package processor

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Taxonomy maps canonical keys to synonym slices.
type Taxonomy map[string][]string

const (
	MaxSynonymsPerCanonical = 20
	MinLabelLen             = 1
	MaxLabelLen             = 50
	MaxExpandedLabels       = 200
	MaxExpandedBytes        = 4 * 1024 * 1024 // 4MB
)

// TaxonomyPath returns filesystem path for taxonomy JSON.
// Env TAXONOMY_PATH overrides default.
func TaxonomyPath() string {
	if p := os.Getenv("TAXONOMY_PATH"); p != "" {
		return p
	}
	return "document_processing_service/config/label_taxonomy.json"
}

// IsFailFast returns true when development fail-fast is enabled.
// Checks FAIL_FAST env, or APP_ENV/ENV == development.
func IsFailFast() bool {
	if v, ok := os.LookupEnv("FAIL_FAST"); ok {
		low := strings.ToLower(strings.TrimSpace(v))
		if low == "true" || low == "1" || low == "yes" {
			return true
		}
		if low == "false" || low == "0" || low == "no" {
			return false
		}
	}
	if env := strings.ToLower(strings.TrimSpace(os.Getenv("APP_ENV"))); env == "development" || env == "dev" {
		return true
	}
	if env := strings.ToLower(strings.TrimSpace(os.Getenv("ENV"))); env == "development" || env == "dev" {
		return true
	}
	return false
}

// IsDebugMode returns true when verbose logging is enabled.
func IsDebugMode() bool {
	if v := strings.ToLower(strings.TrimSpace(os.Getenv("LOG_LEVEL"))); v == "debug" {
		return true
	}
	if v := strings.ToLower(strings.TrimSpace(os.Getenv("OBSERVABILITY"))); v == "verbose" || v == "debug" {
		return true
	}
	return false
}

// LoadTaxonomy reads taxonomy JSON from path, validates entries, and computes SHA256 hash.
func LoadTaxonomy(path string) (Taxonomy, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, "", fmt.Errorf("taxonomy_load_failed: failed to read %s: %w", path, err)
	}
	hash := sha256.Sum256(data)
	hashStr := hex.EncodeToString(hash[:])

	var raw map[string][]string
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, "", fmt.Errorf("taxonomy_load_failed: unparseable JSON %s: %w", path, err)
	}
	if raw == nil {
		raw = make(map[string][]string)
	}
	if err := ValidateTaxonomy(raw); err != nil {
		return nil, "", err
	}
	tax := Taxonomy(raw)
	return tax, hashStr, nil
}

// ValidateTaxonomy checks constraints: key 1-50 chars, each synonym 1-50, <=20 per canonical.
func ValidateTaxonomy(tax map[string][]string) error {
	for k, syns := range tax {
		if len(k) < MinLabelLen || len(k) > MaxLabelLen {
			return fmt.Errorf("taxonomy_load_failed: canonical key %q length %d out of [%d,%d]", k, len(k), MinLabelLen, MaxLabelLen)
		}
		if len(syns) > MaxSynonymsPerCanonical {
			return fmt.Errorf("taxonomy_load_failed: canonical %q has %d synonyms exceeds %d", k, len(syns), MaxSynonymsPerCanonical)
		}
		for _, s := range syns {
			if len(s) < MinLabelLen || len(s) > MaxLabelLen {
				return fmt.Errorf("taxonomy_load_failed: synonym %q for %q length %d out of [%d,%d]", s, k, len(s), MinLabelLen, MaxLabelLen)
			}
		}
	}
	return nil
}

// MustLoadTaxonomy loads taxonomy or panics in fail-fast mode, otherwise returns error.
// Used at startup to satisfy health check fail-fast contract.
func MustLoadTaxonomy(path string) (Taxonomy, string) {
	tax, hash, err := LoadTaxonomy(path)
	if err != nil {
		if IsFailFast() {
			panic(fmt.Sprintf("taxonomy_load_failed: %v", err))
		}
		// In non-fail-fast, return empty taxonomy and hash to allow degraded operation.
		return make(Taxonomy), ""
	}
	return tax, hash
}
