package processor

import (
	"maps"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/AndrewK4758/shared_utils/jsonclean"
)

// Pre-compiled regex for money parsing
var moneyRe = regexp.MustCompile(`\$?\s*([\d,]+\.?\d*)`)

var (
	valueNormTotalMu sync.Mutex
	valueNormTotals  = make(map[string]int)
)

// IncrementValueNormTotal increments metric counter for normalization type
func IncrementValueNormTotal(typ string) {
	valueNormTotalMu.Lock()
	defer valueNormTotalMu.Unlock()
	valueNormTotals[typ]++
}

// GetValueNormTotals returns copy of counters (for testing)
func GetValueNormTotals() map[string]int {
	valueNormTotalMu.Lock()
	defer valueNormTotalMu.Unlock()
	cp := make(map[string]int, len(valueNormTotals))
	maps.Copy(cp, valueNormTotals)
	return cp
}

// ResetValueNormTotals clears counters (for testing)
func ResetValueNormTotals() {
	valueNormTotalMu.Lock()
	defer valueNormTotalMu.Unlock()
	valueNormTotals = make(map[string]int)
}

// NormalizeValue canonicalizes span text based on canonical key type.
// Money: "$10"/"$10.00" -> "1000" cents
// Date: "12/24/2025"/"24/12/2025"/"12/24/2025 10:00am" -> "2025-12-24"
// Label-key (file_number etc): returns raw unchanged
func NormalizeValue(text, canonical string) (string, error) {
	text = jsonclean.SanitizeString(text)
	trimmed := strings.TrimSpace(text)
	canonLower := strings.ToLower(strings.TrimSpace(canonical))

	// Determine type by canonical pattern
	isMoney := isMoneyCanonical(canonLower)
	isDate := isDateCanonical(canonLower)

	if isMoney {
		norm, err := normalizeMoney(trimmed)
		if err != nil {
			if IsFailFast() {
				return "", fmt.Errorf("IMPOSSIBLE_STATE value_normalizer.post_validation_failed: money normalize failed for %q canonical %q: %w", text, canonical, err)
			}
			return "", err
		}
		if norm == "" {
			if IsFailFast() {
				return "", fmt.Errorf("IMPOSSIBLE_STATE value_normalizer.post_validation_failed: empty normalized_value for money canonical %q", canonical)
			}
			return "", fmt.Errorf("empty money normalization for %q", text)
		}
		IncrementValueNormTotal("money")
		return norm, nil
	}
	if isDate {
		norm, err := normalizeDate(trimmed)
		if err != nil {
			if IsFailFast() {
				return "", fmt.Errorf("IMPOSSIBLE_STATE value_normalizer.post_validation_failed: date normalize failed for %q canonical %q: %w", text, canonical, err)
			}
			return "", err
		}
		if norm == "" {
			if IsFailFast() {
				return "", fmt.Errorf("IMPOSSIBLE_STATE value_normalizer.post_validation_failed: empty normalized_value for date canonical %q", canonical)
			}
			return "", fmt.Errorf("empty date normalization for %q", text)
		}
		IncrementValueNormTotal("date")
		return norm, nil
	}
	// Label-key type: preserve raw
	IncrementValueNormTotal("label_key")
	return trimmed, nil
}

func isMoneyCanonical(c string) bool {
	if strings.HasSuffix(c, "_amount") || strings.HasSuffix(c, "_price") || strings.HasSuffix(c, "_rate") {
		// Treat _rate keys (e.g. interest_rate) as money-like after stripping trailing '%'
		return true
	}
	if strings.Contains(c, "amount") || strings.Contains(c, "money") || strings.Contains(c, "price") || strings.Contains(c, "principal") || strings.Contains(c, "loan_amt") {
		return true
	}
	if c == "loan_amount" || c == "interest_rate" || c == "principal" {
		return true
	}
	return false
}

func isDateCanonical(c string) bool {
	if strings.HasSuffix(c, "_date") || strings.Contains(c, "date") {
		return true
	}
	if c == "closing_date" || c == "effective_date" || c == "disbursement_date" {
		return true
	}
	return false
}

func normalizeMoney(s string) (string, error) {
	if s == "" {
		return "", fmt.Errorf("empty money string")
	}
	cleaned := strings.ReplaceAll(s, "$", "")
	cleaned = strings.ReplaceAll(cleaned, ",", "")
	cleaned = strings.TrimSpace(cleaned)
	cleaned = strings.TrimSuffix(cleaned, "%")
	cleaned = strings.TrimSpace(cleaned)

	lower := strings.ToLower(cleaned)
	if before, ok := strings.CutSuffix(lower, "k"); ok {
		numPart := before
		f, err := strconv.ParseFloat(strings.TrimSpace(numPart), 64)
		if err != nil {
			return "", err
		}
		cents := int64(math.Round(f * 1000 * 100))
		return fmt.Sprintf("%d", cents), nil
	}

	m := moneyRe.FindStringSubmatch("$" + cleaned)
	if m == nil {
		f, err := strconv.ParseFloat(cleaned, 64)
		if err != nil {
			return "", fmt.Errorf("money parse failed for %q", s)
		}
		cents := int64(math.Round(f * 100))
		return fmt.Sprintf("%d", cents), nil
	}
	numStr := strings.ReplaceAll(m[1], ",", "")
	numStr = strings.TrimSpace(numStr)
	if numStr == "" {
		return "", fmt.Errorf("money parse empty for %q", s)
	}
	f, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return "", err
	}
	cents := int64(math.Round(f * 100))
	return fmt.Sprintf("%d", cents), nil
}

func normalizeDate(s string) (string, error) {
	if s == "" {
		return "", fmt.Errorf("empty date string")
	}
	input := strings.TrimSpace(s)

	// Try both MDY and DMY layouts with case-insensitive AM/PM handling
	candidates := []string{input, strings.ToLower(input), strings.ToUpper(input)}

	mdyLayouts := []string{
		"01/02/2006 3:04pm",
		"01/02/2006 03:04pm",
		"1/2/2006 3:04pm",
		"1/2/2006 03:04pm",
		"01/02/2006 3:04PM",
		"01/02/2006 03:04PM",
		"1/2/2006 3:04PM",
		"01/02/2006",
		"1/2/2006",
		"2006-01-02",
		"2006/01/02",
	}
	dmyLayouts := []string{
		"02/01/2006 3:04pm",
		"02/01/2006 03:04pm",
		"2/1/2006 3:04pm",
		"02/01/2006",
		"2/1/2006",
	}

	var mdyTime *time.Time
	var dmyTime *time.Time

	for _, cand := range candidates {
		for _, l := range mdyLayouts {
			if t, err := time.Parse(l, cand); err == nil {
				mdyTime = &t
				break
			}
		}
		if mdyTime != nil {
			break
		}
	}
	for _, cand := range candidates {
		for _, l := range dmyLayouts {
			if t, err := time.Parse(l, cand); err == nil {
				dmyTime = &t
				break
			}
		}
		if dmyTime != nil {
			break
		}
	}

	// Flag dates that parse under both MDY and DMY to different calendar days (e.g. 01/02/2025)
	if mdyTime != nil && dmyTime != nil {
		mdyISO := mdyTime.Format("2006-01-02")
		dmyISO := dmyTime.Format("2006-01-02")
		if mdyISO != dmyISO {
			if IsFailFast() {
				return "", fmt.Errorf("DECISION_DRIFT date_ambiguous: 01/02/2025 ambiguous MDY=%s DMY=%s", mdyISO, dmyISO)
			}
		}
		return mdyTime.Format("2006-01-02"), nil
	}
	if mdyTime != nil {
		return mdyTime.Format("2006-01-02"), nil
	}
	if dmyTime != nil {
		return dmyTime.Format("2006-01-02"), nil
	}
	return "", fmt.Errorf("date parse failed for %q", s)
}
