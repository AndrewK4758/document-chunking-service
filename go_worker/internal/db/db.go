package db

import (
	"database/sql"
    "errors"
    "fmt"
	"os"
	"path/filepath"

	"github.com/AndrewK4758/shared_utils/envutil"
	"github.com/AndrewK4758/shared_utils/logger"
	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	_ "github.com/golang-migrate/migrate/v4/source/file"
	_ "github.com/lib/pq"
	gormpostgres "gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func InitDB() *gorm.DB {
	dsn := envutil.MustGetString("DB_CONNECTION_STRING")
	driver := envutil.GetStringOrDefault("DB_DRIVER", "postgres")
	dbname := envutil.GetStringOrDefault("DB_NAME", "doc_processor_service")

	dbSql, err := sql.Open(driver, dsn)
	if err != nil {
		panic(fmt.Errorf("failed to open database: %w", err))
	}

	dbDriver, err := postgres.WithInstance(dbSql, &postgres.Config{})
	if err != nil {
		panic(fmt.Errorf("failed to create postgres driver instance: %w", err))
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
			migrationsDir = "db/migrations"
		}
	}
		migrationsDirAbs, err := filepath.Abs(migrationsDir)
	if err != nil {
		panic(fmt.Errorf("failed to get absolute path for migrations: %w", err))
	}

	m, err := migrate.NewWithDatabaseInstance(
		fmt.Sprintf("file://%s", migrationsDirAbs),
		dbname,
		dbDriver,
	)
	if err != nil {
		panic(fmt.Errorf("failed to initialize migrate instance: %w", err))
	}

	err = m.Up()
	if err != nil && !errors.Is(err, migrate.ErrNoChange) {
		panic(fmt.Errorf("failed to run migrations: %w", err))
	}

	logger.Info("DB", "Successfully ran database migrations.")

	gormDB, err := gorm.Open(gormpostgres.Open(dsn), &gorm.Config{})
	if err != nil {
		panic(fmt.Errorf("failed to connect to gorm: %w", err))
	}

	return gormDB
}
