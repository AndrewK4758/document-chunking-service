package db

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"

	"github.com/AndrewK4758/shared_utils/envutil"
	"github.com/AndrewK4758/shared_utils/logger"
	"github.com/golang-migrate/migrate/v4"
	migrate_postgres "github.com/golang-migrate/migrate/v4/database/postgres"
	_ "github.com/golang-migrate/migrate/v4/source/file"
	gorm_postgres "gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// InitDB initializes the database connection and runs migrations.
func InitDB() *gorm.DB {
	host := envutil.MustGetString("DB_HOST")
	port := envutil.GetStringOrDefault("DB_PORT", "5432")
	user := envutil.MustGetString("DB_USER")
	password := envutil.MustGetString("DB_PASSWORD")
	dbname := envutil.MustGetString("DB_NAME")

	dsn := fmt.Sprintf("host=%s user=%s password=%s dbname=%s port=%s sslmode=disable TimeZone=UTC",
		host, user, password, dbname, port)

	// We use standard database/sql for migrations
	sqlDB, err := sql.Open("postgres", dsn)
	if err != nil {
		panic(fmt.Errorf("failed to open sql connection for migrations: %w", err))
	}

	driver, err := migrate_postgres.WithInstance(sqlDB, &migrate_postgres.Config{})
	if err != nil {
		panic(fmt.Errorf("failed to create migrate postgres driver: %w", err))
	}

	migrationsDir := envutil.GetStringOrDefault("MIGRATIONS_DIR", "")
	if migrationsDir == "" {
		fallbackPaths := []string{
			"db/migrations",
			"go_worker/db/migrations",
			"document_processing_service/go_worker/db/migrations",
			"../db/migrations",
			"../../db/migrations",
		}
		for _, p := range fallbackPaths {
			if _, err := os.Stat(p); err == nil {
				migrationsDir = p
				break
			}
		}
		if migrationsDir == "" {
			migrationsDir = "db/migrations" // ultimate fallback
		}
	}
	
	migrationsDirAbs, err := filepath.Abs(migrationsDir)
	if err != nil {
		panic(fmt.Errorf("failed to get absolute path for migrations: %w", err))
	}

	m, err := migrate.NewWithDatabaseInstance(
		fmt.Sprintf("file://%s", migrationsDirAbs),
		dbname,
		driver,
	)
	if err != nil {
		panic(fmt.Errorf("failed to initialize migrate instance: %w", err))
	}

	err = m.Up()
	if err != nil && err != migrate.ErrNoChange {
		panic(fmt.Errorf("failed to run migrations: %w", err))
	}

	logger.Info("DB", "Successfully ran database migrations.")

	// Connect GORM
	gormDB, err := gorm.Open(gorm_postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		panic(fmt.Errorf("failed to connect to gorm: %w", err))
	}

	return gormDB
}
