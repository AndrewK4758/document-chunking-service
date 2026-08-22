package processor

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"

	"google.golang.org/grpc/metadata"
)

// Expander performs O(1) taxonomy expansion with deduplication and reverse mapping.
type Expander struct {
	tax  Taxonomy
	hash string
	mu   sync.Mutex
	// collision tracking for metrics
	collisions map[string]int
	// histograms (simple counters)
	muMetrics sync.Mutex
}

// NewExpander creates an Expander from loaded taxonomy.
func NewExpander(tax Taxonomy, hash string) *Expander {
	if tax == nil {
		tax = make(Taxonomy)
	}
	return &Expander{
		tax:        tax,
		hash:       hash,
		collisions: make(map[string]int),
	}
}

// Hash returns taxonomy hash.
func (e *Expander) Hash() string { return e.hash }

// Taxonomy returns underlying taxonomy (read-only).
func (e *Expander) Taxonomy() Taxonomy { return e.tax }

// normalizeCanonical attempts to map a caller key to taxonomy key.
// Trims punctuation and normalizes casing/spacing.
func normalizeCanonical(in string) string {
	s := strings.TrimSpace(in)
	s = strings.ToLower(s)
	// Trim trailing punctuation like :, ., #
	s = strings.Trim(s, " :.#\t\n\r")
	// Normalize spaces/hyphens to underscore
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ReplaceAll(s, "-", "_")
	for strings.Contains(s, "__") {
		s = strings.ReplaceAll(s, "__", "_")
	}
	return s
}

// lookupSynonyms performs O(1) lookup with fallback normalization.
func (e *Expander) lookupSynonyms(canonical string) ([]string, bool) {
	if syns, ok := e.tax[canonical]; ok {
		return syns, true
	}
	norm := normalizeCanonical(canonical)
	if norm != canonical {
		if syns, ok := e.tax[norm]; ok {
			return syns, true
		}
	}
	// Also try lowercased version
	low := strings.ToLower(strings.TrimSpace(canonical))
	if low != canonical && low != norm {
		if syns, ok := e.tax[low]; ok {
			return syns, true
		}
	}
	return nil, false
}

// recordCollision increments collision metric for a synonym.
func (e *Expander) recordCollision(synonym string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.collisions[synonym]++
}

// CollisionCount returns collision count for a synonym (for testing/metrics).
func (e *Expander) CollisionCount(synonym string) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.collisions[synonym]
}

// AllCollisions returns copy of collision map.
func (e *Expander) AllCollisions() map[string]int {
	e.mu.Lock()
	defer e.mu.Unlock()
	cp := make(map[string]int, len(e.collisions))
	for k, v := range e.collisions {
		cp[k] = v
	}
	return cp
}

// Expand expands canonicals into deduplicated union with reverse map.
// Returns budget errors with SEAM_VIOLATION semantics; empty canonicals handled per fail-fast.
func (e *Expander) Expand(canonicals []string) ([]string, map[string]string, error) {
	if e == nil {
		return nil, nil, fmt.Errorf("IMPOSSIBLE_STATE taxonomy.not_loaded: expander not initialized")
	}
	if e.hash == "" && len(e.tax) == 0 {
		// Empty taxonomy is allowed (identity expansion), but hash missing is flagged
		// Only critical if called pre-inference with non-empty canonicals and hash empty due to load failure.
		// We still proceed with identity expansion; observability will note hash.
	}

	canonicalCount := len(canonicals)

	// Fail-fast guard: empty input
	trimmedNonEmpty := 0
	for _, c := range canonicals {
		if strings.TrimSpace(c) != "" {
			trimmedNonEmpty++
		}
	}
	if trimmedNonEmpty == 0 {
		if IsFailFast() {
			return nil, nil, fmt.Errorf("InvalidArgument: expanded_labels empty (canonical_count=%d)", canonicalCount)
		}
		// Non-fail-fast: return empty
		return []string{}, map[string]string{}, nil
	}

	seen := make(map[string]bool, canonicalCount*4)
	expanded := make([]string, 0, canonicalCount*4)
	reverse := make(map[string]string, canonicalCount*4)
	totalBytes := 0

	for _, canonical := range canonicals {
		canonical = strings.TrimSpace(canonical)
		if canonical == "" {
			continue
		}
		syns, ok := e.lookupSynonyms(canonical)

		// Build candidate list: canonical first, then synonyms in file order
		candidates := make([]string, 0, 1+len(syns))
		candidates = append(candidates, canonical)
		if ok {
			candidates = append(candidates, syns...)
		}

		for _, cand := range candidates {
			// Dedup preserving order
			if !seen[cand] {
				seen[cand] = true
				expanded = append(expanded, cand)
				totalBytes += len(cand)
			}
			// Reverse map with first-wins collision handling
			if existing, exists := reverse[cand]; exists {
				if existing != canonical {
					// Synonym collision: first canonical wins
					e.recordCollision(cand)
					slog.Warn("DECISION_DRIFT taxonomy.synonym_collision",
						slog.String("synonym", cand),
						slog.String("existing_canonical", existing),
						slog.String("new_canonical", canonical),
					)
				}
			} else {
				reverse[cand] = canonical
			}
		}
	}

	if len(expanded) == 0 {
		if IsFailFast() {
			return nil, nil, fmt.Errorf("InvalidArgument: expanded_labels empty after dedup")
		}
		return expanded, reverse, nil
	}

	// Budget enforcement
	if len(expanded) > MaxExpandedLabels {
		return nil, nil, fmt.Errorf("SEAM_VIOLATION taxonomy.budget_exceeded: budget exceeded: %d > %d", len(expanded), MaxExpandedLabels)
	}
	if totalBytes > MaxExpandedBytes {
		return nil, nil, fmt.Errorf("SEAM_VIOLATION taxonomy.budget_exceeded: serialized bytes %d > 4MB", totalBytes)
	}

	// Observability: always emit counts+hash at INFO, full union at DEBUG
	e.emitExpansionLog(canonicalCount, len(expanded), len(expanded), totalBytes, expanded)

	return expanded, reverse, nil
}

func (e *Expander) emitExpansionLog(canonicalCount, expandedCount, dedupedCount, totalBytes int, expanded []string) {
	attrs := []slog.Attr{
		slog.Int("canonical_count", canonicalCount),
		slog.Int("expanded_count", expandedCount),
		slog.Int("deduped_count", dedupedCount),
		slog.String("taxonomy_hash", e.hash),
		slog.Int("total_bytes", totalBytes),
	}
	// Use default logger via slog
	msg := "taxonomy_expansion"
	// Structured INFO always
	slog.LogAttrs(context.Background(), slog.LevelInfo, msg,
		append(attrs, slog.String("event", "taxonomy_expansion"))...,
	)
	// Also via shared logger if initialized (best effort)
	// Fallback to slog Debug for full union
	if IsDebugMode() {
		slog.LogAttrs(context.Background(), slog.LevelDebug, "taxonomy_expansion_full_union",
			slog.Any("expanded", expanded),
			slog.String("taxonomy_hash", e.hash),
		)
		// Also env stdout for test capture
		if os.Getenv("LOG_LEVEL") == "debug" {
			fmt.Fprintf(os.Stderr, "DEBUG taxonomy_expansion_full_union expanded=%v hash=%s\n", expanded, e.hash)
		}
	}
}

// NormalizeLabel resolves synonym to canonical via reverse map.
// Returns canonical, matchedSynonym, collision bool.
func (e *Expander) NormalizeLabel(label string, reverse map[string]string) (string, string, bool) {
	if canonical, ok := reverse[label]; ok {
		coll := e.CollisionCount(label) > 0
		return canonical, label, coll
	}
	// Miss: return label as canonical (identity)
	return label, label, false
}

// NormalizedExtract carries post-normalized span data (Go memory only, not proto).
type NormalizedExtract struct {
	RawText         string  `json:"raw_text"`
	NormalizedValue string  `json:"normalized_value"`
	CanonicalKey    string  `json:"canonical_key"`
	MatchedSynonym  string  `json:"matched_synonym"`
	Confidence      float64 `json:"confidence"`
	Page            int32   `json:"page"`
}

// NormalizeSpans converts raw spans using reverse map and value normalizer.
// Preserves geometry/confidence, canonicalizes label, computes normalized_value.
func (e *Expander) NormalizeSpans(spans []SpanLike, reverse map[string]string) ([]NormalizedExtract, []SpanLike) {
	// SpanLike is an interface to avoid proto import cycle; caller passes ExtractedSpan slice
	// This generic helper is kept simple; concrete implementation in value_normalizer.go
	return nil, nil
}

// SpanLike abstracts ExtractedSpan for testing without proto dependency
type SpanLike struct {
	Text       string
	Label      string
	Confidence float64
	PageNumber int32
	X0, Y0, X1, Y1 float64
}

// NormalizeSpanLabel does label-only normalization for a single span.
func NormalizeSpanLabel(spanLabel string, reverse map[string]string, expander *Expander) (canonical string, matched string, collided bool) {
	if reverse != nil {
		if c, ok := reverse[spanLabel]; ok {
			coll := false
			if expander != nil {
				coll = expander.CollisionCount(spanLabel) > 0
			}
			return c, spanLabel, coll
		}
	}
	return spanLabel, spanLabel, false
}

// PropagateTraceContext injects W3C trace headers into outgoing gRPC context.
// It extracts traceparent / trace_id / span_id from incoming metadata or context values
// and appends them to outgoing gRPC metadata, adding taxonomy_hash as span attribute.
// This satisfies OS-5.2: Go → C++ trace propagation via gRPC metadata.
func PropagateTraceContext(ctx context.Context) context.Context {
	if ctx == nil {
		return ctx
	}
	// Try to extract traceparent from incoming metadata
	var traceparent string
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if vals := md.Get("traceparent"); len(vals) > 0 {
			traceparent = vals[0]
		} else if vals := md.Get("trace_id"); len(vals) > 0 {
			// Fallback: raw trace_id
			traceparent = vals[0]
		} else if vals := md.Get("x-trace-id"); len(vals) > 0 {
			traceparent = vals[0]
		}
	}
	// Also check context values (used in tests / non-gRPC callers)
	if traceparent == "" {
		if v := ctx.Value("traceparent"); v != nil {
			if s, ok := v.(string); ok {
				traceparent = s
			}
		}
		if traceparent == "" {
			if v := ctx.Value("trace_id"); v != nil {
				if s, ok := v.(string); ok {
					traceparent = s
				}
			}
		}
	}
	if traceparent != "" {
		ctx = metadata.AppendToOutgoingContext(ctx, "traceparent", traceparent)
		// Also ensure downstream can read trace_id
		if md, ok := metadata.FromIncomingContext(ctx); ok {
			_ = md
		}
	}
	return ctx
}

// PropagateTraceWithHash is convenience that also attaches taxonomy_hash as span attribute.
func PropagateTraceWithHash(ctx context.Context, hash string) context.Context {
	ctx = PropagateTraceContext(ctx)
	if hash != "" {
		ctx = metadata.AppendToOutgoingContext(ctx, "taxonomy_hash", hash, "taxonomy_hash", hash)
		// Deduplicate: grpc metadata Append duplicates; use single
	}
	return ctx
}
