package db

import (
	"time"
)

// MlMetric represents a tracked execution step in the ML pipeline.
type MlMetric struct {
	TenantID     string     `gorm:"column:tenant_id;primaryKey;not null"`
	AppID        string     `gorm:"column:app_id;primaryKey;not null"`
	JobID        string     `gorm:"column:job_id;primaryKey;not null"`
	StepName     string     `gorm:"column:step_name;primaryKey;not null"`
	Status       string     `gorm:"column:status;not null"`
	ErrorMessage string     `gorm:"column:error_message"`
	StartedAt    time.Time  `gorm:"column:started_at;not null"`
	CompletedAt  *time.Time `gorm:"column:completed_at"`
}

// TableName overrides the default table name to utilize a specific schema
func (MlMetric) TableName() string {
	return "ml_metrics.ml_metric_executions"
}
