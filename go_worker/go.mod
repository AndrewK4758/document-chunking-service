module document_processing_service

go 1.27.0

require (
	github.com/AndrewK4758/shared_protos v0.0.0-20260824195642-ec565d8fd7a4
	github.com/AndrewK4758/shared_utils v0.0.0-00010101000000-000000000000
	github.com/bodgit/sevenzip v1.6.5
	github.com/golang-migrate/migrate/v4 v4.19.1
	github.com/grpc-ecosystem/go-grpc-middleware/v2 v2.3.3
	github.com/lib/pq v1.12.3
	github.com/mholt/archiver/v3 v3.5.1
	github.com/nats-io/nats.go v1.53.1
	github.com/pdfcpu/pdfcpu v0.15.0
	github.com/stretchr/testify v1.12.1
	golang.org/x/sync v0.22.0
	google.golang.org/grpc v1.83.2
	google.golang.org/protobuf v1.36.12
	gorm.io/driver/postgres v1.6.2
	gorm.io/gorm v1.31.2
)

require (
	github.com/andybalholm/brotli v1.2.2 // indirect
	github.com/bodgit/plumbing v1.3.0 // indirect
	github.com/bodgit/windows v1.0.1 // indirect
	github.com/clipperhouse/uax29/v2 v2.7.0 // indirect
	github.com/docker/go-connections v0.8.1 // indirect
	github.com/dsnet/compress v0.0.2-0.20210315054119-f66993602bf5 // indirect
	github.com/fsnotify/fsnotify v1.10.1 // indirect
	github.com/go-viper/mapstructure/v2 v2.5.0 // indirect
	github.com/golang/snappy v1.0.0 // indirect
	github.com/hashicorp/golang-lru/v2 v2.0.7 // indirect
	github.com/hhrutter/tiff v1.0.6 // indirect
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/pgx/v5 v5.10.0 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
	github.com/jinzhu/inflection v1.0.0 // indirect
	github.com/jinzhu/now v1.1.5 // indirect
	github.com/klauspost/compress v1.19.2 // indirect
	github.com/klauspost/pgzip v1.2.6 // indirect
	github.com/mattn/go-runewidth v0.0.27 // indirect
	github.com/moby/term v0.5.2 // indirect
	github.com/nats-io/nkeys v0.4.16 // indirect
	github.com/nats-io/nuid v1.0.1 // indirect
	github.com/nwaples/rardecode v1.1.3 // indirect
	github.com/opencontainers/image-spec v1.1.1 // indirect
	github.com/pelletier/go-toml/v2 v2.4.3 // indirect
	github.com/pierrec/lz4/v4 v4.1.29 // indirect
	github.com/sagikazarmark/locafero v0.12.0 // indirect
	github.com/spf13/afero v1.15.0 // indirect
	github.com/spf13/cast v1.10.0 // indirect
	github.com/spf13/pflag v1.0.10 // indirect
	github.com/spf13/viper v1.21.0 // indirect
	github.com/stangelandcl/ppmd v0.1.1 // indirect
	github.com/subosito/gotenv v1.6.0 // indirect
	github.com/ulikunitz/xz v0.5.16 // indirect
	github.com/xi2/xz v0.0.0-20171230120015-48954b6210f8 // indirect
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.70.0 // indirect
	go.opentelemetry.io/otel/metric v1.45.0 // indirect
	go.opentelemetry.io/otel/sdk v1.45.0 // indirect
	go.opentelemetry.io/otel/trace v1.45.0 // indirect
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	go4.org v0.0.0-20260112195520-a5071408f32f // indirect
	golang.org/x/crypto v0.55.0 // indirect
	golang.org/x/image v0.44.0 // indirect
	golang.org/x/net v0.58.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.41.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260819154853-08b0e4226688 // indirect
	gopkg.in/natefinch/lumberjack.v2 v2.2.1 // indirect
)

replace github.com/AndrewK4758/shared_utils => ../../shared_utils

replace github.com/AndrewK4758/shared_protos => ../../shared_protos
