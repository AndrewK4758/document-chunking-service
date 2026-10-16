#pragma once

#include <string>
#include <stdexcept>
#include <grpcpp/grpcpp.h>
#include <grpcpp/support/status.h>
#include <spdlog/spdlog.h>

// Translates C++ exceptions into grpc::Status responses and emits diagnostic logs.

namespace doc_processor {

inline std::string ExtractTraceparent(grpc::ServerContext* context) {
    if (!context) return "";
    auto meta = context->client_metadata();
    auto it = meta.find("traceparent");
    if (it != meta.end()) return std::string(it->second.data(), it->second.size());
    auto it2 = meta.find("grpc-trace-bin");
    if (it2 != meta.end()) return std::string(it2->second.data(), it2->second.size());
    return "";
}

inline std::string FormatDetail(const std::string& job_id, const std::string& what) {
    if (job_id.empty()) return what;
    return "[JOB: " + job_id + "] " + what;
}

template <typename Request>
inline std::string ExtractJobId(const Request* request) {
    if (!request) return "";
    // Assumes Request provides protobuf has_identity() and identity().job_id().
    if (!request->has_identity()) return "";
    return request->identity().job_id();
}

inline std::string ExtractJobIdFromRequest(const std::string& job_id) {
    return job_id;
}

inline void PublishTraceabilityError(
    const std::string& job_id,
    const std::string& tenant_id,
    const std::string& app_id,
    const std::string& traceparent,
    const std::string& method,
    grpc::StatusCode code,
    const std::string& what) {
    // Stubbed until a NATS client is wired into cpp_ml_worker.
    // Publishing errors must not alter the returned gRPC status.
    try {
        (void)job_id;
        (void)tenant_id;
        (void)app_id;
        (void)traceparent;
        (void)method;
        (void)code;
        (void)what;
    } catch (...) {
        try {
            spdlog::warn("Traceability NATS publish failed for job {}", job_id.empty() ? "unknown" : job_id);
        } catch (...) {}
    }
}

inline void EmitDiagnostics(
    const std::string& job_id,
    const std::string& traceparent,
    const std::string& method,
    grpc::StatusCode code,
    const std::string& what,
    const std::string& tenant_id = "",
    const std::string& app_id = "") {
    try {
        const char* code_name = (code == grpc::StatusCode::INVALID_ARGUMENT) ? "INVALID_ARGUMENT" : "INTERNAL";
        spdlog::error("[JOB: {}] gRPC {} failed code={} traceparent={} what={}",
            job_id.empty() ? "unknown" : job_id,
            method,
            code_name,
            traceparent.empty() ? "-" : traceparent,
            what);
    } catch (...) {}
    try {
        PublishTraceabilityError(job_id, tenant_id, app_id, traceparent, method, code, what);
    } catch (...) {
        try {
            spdlog::warn("Traceability NATS publish failed for job {}", job_id.empty() ? "unknown" : job_id);
        } catch (...) {}
    }
}

inline grpc::Status GrpcStatusFromException(
    const std::invalid_argument& ex,
    const std::string& job_id,
    const std::string& traceparent,
    const std::string& method = "CppMlWorkerService/RedactDocument",
    const std::string& tenant_id = "",
    const std::string& app_id = "") {
    std::string detail = FormatDetail(job_id, ex.what());
    EmitDiagnostics(job_id, traceparent, method, grpc::StatusCode::INVALID_ARGUMENT, ex.what(), tenant_id, app_id);
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, detail);
}

inline grpc::Status GrpcStatusFromException(
    const std::out_of_range& ex,
    const std::string& job_id,
    const std::string& traceparent,
    const std::string& method = "CppMlWorkerService/RedactDocument",
    const std::string& tenant_id = "",
    const std::string& app_id = "") {
    std::string detail = FormatDetail(job_id, ex.what());
    EmitDiagnostics(job_id, traceparent, method, grpc::StatusCode::INVALID_ARGUMENT, ex.what(), tenant_id, app_id);
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, detail);
}

inline grpc::Status GrpcStatusFromException(
    const std::exception& ex,
    const std::string& job_id,
    const std::string& traceparent,
    const std::string& method = "CppMlWorkerService/RedactDocument",
    const std::string& tenant_id = "",
    const std::string& app_id = "") {
    std::string detail = FormatDetail(job_id, ex.what());
    EmitDiagnostics(job_id, traceparent, method, grpc::StatusCode::INTERNAL, ex.what(), tenant_id, app_id);
    return grpc::Status(grpc::StatusCode::INTERNAL, detail);
}

inline grpc::Status GrpcStatusFromUnknown(
    const std::string& job_id,
    const std::string& traceparent,
    const std::string& method = "CppMlWorkerService/RedactDocument",
    const std::string& tenant_id = "",
    const std::string& app_id = "") {
    std::string detail = job_id.empty() ? "Unknown exception" : "[JOB: " + job_id + "] Unknown exception";
    EmitDiagnostics(job_id, traceparent, method, grpc::StatusCode::INTERNAL, "Unknown exception", tenant_id, app_id);
    return grpc::Status(grpc::StatusCode::INTERNAL, detail);
}

// Translates the active exception in a catch block into a grpc::Status.
inline grpc::Status GrpcStatusFromCurrentException(
    const std::string& job_id,
    const std::string& traceparent,
    const std::string& method = "CppMlWorkerService/RedactDocument",
    const std::string& tenant_id = "",
    const std::string& app_id = "") {
    try {
        std::rethrow_exception(std::current_exception());
    } catch (const std::invalid_argument& ex) {
        return GrpcStatusFromException(ex, job_id, traceparent, method, tenant_id, app_id);
    } catch (const std::out_of_range& ex) {
        return GrpcStatusFromException(ex, job_id, traceparent, method, tenant_id, app_id);
    } catch (const std::exception& ex) {
        return GrpcStatusFromException(ex, job_id, traceparent, method, tenant_id, app_id);
    } catch (...) {
        return GrpcStatusFromUnknown(job_id, traceparent, method, tenant_id, app_id);
    }
}

template <typename Request, typename Func>
inline grpc::Status WrapGrpc(
    grpc::ServerContext* context,
    const Request* request,
    const std::string& method,
    Func&& fn) {
    try {
        fn();
        return grpc::Status::OK;
    } catch (const std::invalid_argument& ex) {
        std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
        std::string tenant_id = request && request->has_identity() ? request->identity().tenant_id() : "";
        std::string app_id = request && request->has_identity() ? request->identity().app_id() : "";
        std::string traceparent = ExtractTraceparent(context);
        return GrpcStatusFromException(ex, job_id, traceparent, method, tenant_id, app_id);
    } catch (const std::out_of_range& ex) {
        std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
        std::string tenant_id = request && request->has_identity() ? request->identity().tenant_id() : "";
        std::string app_id = request && request->has_identity() ? request->identity().app_id() : "";
        std::string traceparent = ExtractTraceparent(context);
        return GrpcStatusFromException(ex, job_id, traceparent, method, tenant_id, app_id);
    } catch (const std::exception& ex) {
        std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
        std::string tenant_id = request && request->has_identity() ? request->identity().tenant_id() : "";
        std::string app_id = request && request->has_identity() ? request->identity().app_id() : "";
        std::string traceparent = ExtractTraceparent(context);
        return GrpcStatusFromException(ex, job_id, traceparent, method, tenant_id, app_id);
    } catch (...) {
        std::string job_id = request && request->has_identity() ? request->identity().job_id() : "";
        std::string tenant_id = request && request->has_identity() ? request->identity().tenant_id() : "";
        std::string app_id = request && request->has_identity() ? request->identity().app_id() : "";
        std::string traceparent = ExtractTraceparent(context);
        return GrpcStatusFromUnknown(job_id, traceparent, method, tenant_id, app_id);
    }
}

} // namespace doc_processor
