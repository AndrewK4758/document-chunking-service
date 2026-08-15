CREATE SCHEMA IF NOT EXISTS ml_metrics;

CREATE TABLE IF NOT EXISTS ml_metrics.ml_metric_executions (
    tenant_id VARCHAR(255) NOT NULL,
    app_id VARCHAR(255) NOT NULL,
    job_id VARCHAR(255) NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    PRIMARY KEY (tenant_id, app_id, job_id, step_name)
);

CREATE INDEX idx_ml_metrics_job ON ml_metrics.ml_metric_executions (job_id);
