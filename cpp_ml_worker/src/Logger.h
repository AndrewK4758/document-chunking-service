#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace doc_processor {

inline bool LoadEnvFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (size_t pos = line.find('='); pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            if (!value.empty() && value.back() == '\r') {
                value.pop_back();
            }
            setenv(key.c_str(), value.c_str(), 0);
        }
    }
    return true;
}

inline void InitEnv() {
    const std::vector<std::string> search_paths = {
        ".env",
        "../.env",
        "../../.env",
        "../../../.env",
        "document_processing_service/cpp_ml_worker/.env",
        "cpp_ml_worker/.env"
    };
    for (const std::string& path : search_paths) {
        LoadEnvFile(path);
    }
}

inline void InitLogger(const std::string& service_name = "cpp_ml_worker") {
    std::vector<spdlog::sink_ptr> sinks;

    // 1. Console Sink (stdout with ANSI TrueColor level styling)
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    sinks.push_back(console_sink);

    // 2. File Sink (writes directly to target logs/ directory)
    std::string log_file_path = "/home/ak/projects/mployee/logs/cpp_ml_worker.log";
    if (const char* env_path = std::getenv("LOG_FILE_PATH")) {
        if (std::string(env_path).length() > 0) {
            log_file_path = env_path;
        }
    }

    std::filesystem::path p(log_file_path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            throw std::runtime_error("Failed to create log directory '" + p.parent_path().string() + "': " + ec.message());
        }
    }

    // 100 MB max file size, 3 backup files (aligned to Go worker and other services)
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_file_path, 100 * 1024 * 1024, 3);
    file_sink->set_level(spdlog::level::info);
    sinks.push_back(file_sink);

    // 3. Multi-sink Logger Configuration
    auto combined_logger = std::make_shared<spdlog::logger>(service_name, sinks.begin(), sinks.end());
    combined_logger->set_level(spdlog::level::info);
    combined_logger->flush_on(spdlog::level::info);
    combined_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    spdlog::set_default_logger(combined_logger);
}

} // namespace doc_processor
