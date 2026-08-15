package db

import (
	"testing"
	"time"

	"github.com/AndrewK4758/shared_utils/envutil"
	"github.com/stretchr/testify/assert"
)

func TestMlMetricModel(t *testing.T) {
	// Simple struct test to verify tags and types compile
	metric := MlMetric{
		TenantID:     "t1",
		AppID:        "a1",
		JobID:        "j1",
		StepName:     "s1",
		Status:       "STARTED",
		StartedAt:    time.Now(),
	}

	assert.Equal(t, "t1", metric.TenantID)
	assert.Equal(t, "ml_metrics.ml_metric_executions", metric.TableName())
}

func TestInitDB(t *testing.T) {
	// Skip actual DB init if no DB environment variables are set to avoid breaking CI
	if envutil.GetStringOrDefault("DB_HOST", "") == "" {
		t.Skip("Skipping DB test, DB_HOST not set")
	}

	gormDB := InitDB()
	assert.NotNil(t, gormDB)
}
