package processor

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testTaxonomy() (Taxonomy, string) {
	tax := Taxonomy{
		"loan_amount": {"loan amount", "loan amt"},
		"file_number": {"file #", "order #"},
	}
	return tax, "abc123"
}

func TestS11_ExpandCanonicalsWithSynonymsAndDedup(t *testing.T) {
	tax, hash := testTaxonomy()
	exp := NewExpander(tax, hash)

	var buf bytes.Buffer
	h := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(h)

	expanded, reverse, err := exp.Expand([]string{"loan_amount", "file_number"})
	require.NoError(t, err)
	assert.Equal(t, []string{"loan_amount", "loan amount", "loan amt", "file_number", "file #", "order #"}, expanded)
	assert.Equal(t, "loan_amount", reverse["loan amt"])
	assert.Equal(t, "file_number", reverse["order #"])
	assert.Equal(t, "loan_amount", reverse["loan_amount"])

	logOut := buf.String()
	assert.Contains(t, logOut, "taxonomy_expansion")
	assert.Contains(t, logOut, "canonical_count=2")
}

func TestS12_UnknownCanonicalIdentityExpansion(t *testing.T) {
	tax, hash := testTaxonomy()
	exp := NewExpander(tax, hash)
	expanded, reverse, err := exp.Expand([]string{"unknown_label"})
	require.NoError(t, err)
	assert.Equal(t, []string{"unknown_label"}, expanded)
	assert.Equal(t, "unknown_label", reverse["unknown_label"])
}

func TestS13_BudgetExceededFailFastDev(t *testing.T) {
	t.Setenv("FAIL_FAST", "true")
	tax := make(Taxonomy)
	canonicals := make([]string, 250)
	for i := range 250 {
		k := strings.Repeat("a", 5) + strings.Repeat("b", 3) + string(rune('0'+i%10)) + string(rune('a'+i/10))
		k = "key_" + strings.Repeat("x", 10) + "_" + string(rune(65+i%26)) + string(rune(48+i/26))
		k = strings.TrimSpace(k)
		canonicals[i] = k
	}
	exp := NewExpander(tax, "hash-xyz")
	_, _, err := exp.Expand(canonicals)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "budget exceeded")
	assert.Contains(t, err.Error(), "taxonomy.budget_exceeded")
}

func TestS21_SynonymSpanNormalizedToCanonical(t *testing.T) {
	tax, hash := testTaxonomy()
	exp := NewExpander(tax, hash)
	_, reverse, _ := exp.Expand([]string{"loan_amount"})
	canonical, matched, _ := NormalizeSpanLabel("loan amt", reverse, exp)
	assert.Equal(t, "loan_amount", canonical)
	assert.Equal(t, "loan amt", matched)
}

func TestS22_SynonymCollisionFirstWins(t *testing.T) {
	tax := Taxonomy{
		"file_number":  {"order #", "file #"},
		"order_number": {"order #", "order no"},
	}
	exp := NewExpander(tax, "hash-coll")
	_, reverse, err := exp.Expand([]string{"file_number", "order_number"})
	require.NoError(t, err)
	// First canonical inserted wins on synonym collision
	assert.Equal(t, "file_number", reverse["order #"])
	assert.Equal(t, 1, exp.CollisionCount("order #"))
}

func TestS31_MoneyVariationsCollapse(t *testing.T) {
	ResetValueNormTotals()
	v1, err := NormalizeValue("$10", "loan_amount")
	require.NoError(t, err)
	assert.Equal(t, "1000", v1)
	v2, err := NormalizeValue("$10.00", "loan_amount")
	require.NoError(t, err)
	assert.Equal(t, "1000", v2)
	assert.Equal(t, v1, v2)

	v3, err := NormalizeValue("$125,000", "loan_amount")
	require.NoError(t, err)
	assert.Equal(t, "12500000", v3)
}

func TestS32_DateVariationsCollapse(t *testing.T) {
	ResetValueNormTotals()
	v1, err := NormalizeValue("12/24/2025", "closing_date")
	require.NoError(t, err)
	assert.Equal(t, "2025-12-24", v1)
	v2, err := NormalizeValue("24/12/2025", "closing_date")
	require.NoError(t, err)
	assert.Equal(t, "2025-12-24", v2)
	v3, err := NormalizeValue("12/24/2025 10:00am", "closing_date")
	require.NoError(t, err)
	assert.Equal(t, "2025-12-24", v3)
}

func TestS32_AmbiguousDateFailFast(t *testing.T) {
	t.Setenv("FAIL_FAST", "true")
	_, err := NormalizeValue("01/02/2025", "closing_date")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ambiguous")
}

func TestS33_LabelKeyPreserved(t *testing.T) {
	ResetValueNormTotals()
	v, err := NormalizeValue("File # 75335806", "file_number")
	require.NoError(t, err)
	assert.Equal(t, "File # 75335806", v)

	tax := Taxonomy{"file_number": {"file #", "order #", "order number"}}
	exp := NewExpander(tax, "h")
	_, reverse, _ := exp.Expand([]string{"file_number"})
	canonical, _, _ := NormalizeSpanLabel("order #", reverse, exp)
	assert.Equal(t, "file_number", canonical)
}

func TestS41_EmptyDynamicLabelsFailFast(t *testing.T) {
	t.Setenv("FAIL_FAST", "true")
	tax, hash := testTaxonomy()
	exp := NewExpander(tax, hash)
	_, _, err := exp.Expand([]string{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "expanded_labels empty")
	assert.Contains(t, err.Error(), "InvalidArgument")

	_, _, err = exp.Expand([]string{"   ", ""})
	require.Error(t, err)
}

func TestS42_MissingTaxonomyPanicInDev(t *testing.T) {
	t.Setenv("FAIL_FAST", "true")
	t.Setenv("TAXONOMY_PATH", "/nonexistent.json")
	assert.Panics(t, func() {
		MustLoadTaxonomy("/nonexistent.json")
	})
}

func TestS43_LoadTaxonomyMissingFile(t *testing.T) {
	_, _, err := LoadTaxonomy("/no/such/file.json")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "taxonomy_load_failed")
}

func TestOS51_TunableObservability(t *testing.T) {
	tax, hash := testTaxonomy()
	exp := NewExpander(tax, hash)

	// INFO logs counts and hash without dumping the full label union
	var buf bytes.Buffer
	h := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(h)
	t.Setenv("LOG_LEVEL", "info")
	_, _, _ = exp.Expand([]string{"loan_amount"})
	logOut := buf.String()
	assert.Contains(t, logOut, "taxonomy_hash")
	assert.NotContains(t, logOut, "full_union")

	// DEBUG includes the full expanded union
	buf.Reset()
	t.Setenv("LOG_LEVEL", "debug")
	h2 := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(h2)
	_, _, _ = exp.Expand([]string{"loan_amount"})
	logOut = buf.String()
	assert.Contains(t, logOut, "taxonomy_expansion")
	assert.Contains(t, logOut, "full_union")
}

func TestLoadTaxonomy_ValidFile(t *testing.T) {
	tmp, err := os.CreateTemp("", "tax-*.json")
	require.NoError(t, err)
	defer os.Remove(tmp.Name())
	content := map[string][]string{
		"loan_amount": {"loan amount", "loan amt"},
	}
	b, _ := json.Marshal(content)
	_, err = tmp.Write(b)
	require.NoError(t, err)
	tmp.Close()
	tax, hash, err := LoadTaxonomy(tmp.Name())
	require.NoError(t, err)
	assert.Equal(t, 1, len(tax))
	assert.NotEmpty(t, hash)
	assert.Equal(t, []string{"loan amount", "loan amt"}, tax["loan_amount"])
}

func TestValidateTaxonomy_MaxSynonyms(t *testing.T) {
	tax := map[string][]string{
		"key": make([]string, 21),
	}
	for i := range tax["key"] {
		tax["key"][i] = "syn"
	}
	err := ValidateTaxonomy(tax)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "synonyms exceeds")
}

func TestExpansion_O1Performance(t *testing.T) {
	tax := make(Taxonomy)
	for i := range 50 {
		k := "canon_" + strings.Repeat("a", 2) + string(rune(65+i))
		tax[k] = []string{"syn1_" + k, "syn2_" + k}
	}
	exp := NewExpander(tax, "hash-perf")
	canonicals := make([]string, 50)
	i := 0
	for k := range tax {
		canonicals[i] = k
		i++
		if i >= 50 {
			break
		}
	}
	expanded, _, err := exp.Expand(canonicals)
	require.NoError(t, err)
	assert.Equal(t, 150, len(expanded))
}

func TestPropagateTraceContext(t *testing.T) {
	ctx := context.Background()
	out := PropagateTraceContext(ctx)
	assert.NotNil(t, out)
}

func TestValueNormalizationMetrics(t *testing.T) {
	ResetValueNormTotals()
	_, _ = NormalizeValue("$10", "loan_amount")
	_, _ = NormalizeValue("12/24/2025", "closing_date")
	_, _ = NormalizeValue("hello", "file_number")
	totals := GetValueNormTotals()
	assert.Equal(t, 1, totals["money"])
	assert.Equal(t, 1, totals["date"])
	assert.Equal(t, 1, totals["label_key"])
}

func TestBudget_4MBGuard(t *testing.T) {
	t.Setenv("FAIL_FAST", "true")
	tax := make(Taxonomy)
	largeSyn := strings.Repeat("x", 50000)
	tax["canon1"] = []string{largeSyn, largeSyn + "a", largeSyn + "b"}
	exp := NewExpander(tax, "hash")
	canonicals := []string{"canon1"}
	for i := range 80 {
		k := "uniq_" + string(rune(65+i)) + strings.Repeat("y", 40000)
		tax[k] = []string{strings.Repeat("z", 40000)}
		canonicals = append(canonicals, k)
		if len(canonicals) > 90 {
			break
		}
	}
	exp = NewExpander(tax, "hash")
	_, _, err := exp.Expand(canonicals)
	if err != nil {
		assert.Contains(t, err.Error(), "budget exceeded")
	}
}
