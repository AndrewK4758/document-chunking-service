#pragma once

#include <string>
#include <stdexcept>
#include <grpcpp/grpcpp.h>
#include <grpcpp/support/status.h>
#include <spdlog/spdlog.h>

// GrpcStatusHelper — reusable typed exception → grpc::Status translation
// covering every Service:: override in cpp_ml_worker (AC-4). Header-only so
// tests can call GrpcStatusFromException directly without a live server.
// Dual-sink diagnostics: spdlog::error to stderr + best-effort Traceability
// NATS publish (no-op until NATS client injected, swallow-on-failure per AC-3).

namespace doc_processor {

// ---------------------------------------------------------------------------
// Traceparent extraction (AC-3)
// ---------------------------------------------------------------------------
inline std::string ExtractTraceparent(grpc::ServerContext* context) {
    if (!context) return "";
    auto meta = context->client_metadata();
    auto it = meta.find("traceparent");
    if (it != meta.end()) return std::string(it->second.data(), it->second.size());
    auto it2 = meta.find("grpc-trace-bin");
    if (it2 != meta.end()) return std::string(it2->second.data(), it2->second.size());
    return "";
}

// ---------------------------------------------------------------------------
// Detail formatting — embeds [JOB: job_id] when present (AC-1, AC-3)
// ---------------------------------------------------------------------------
inline std::string FormatDetail(const std::string& job_id, const std::string& what) {
    if (job_id.empty()) return what;
    return "[JOB: " + job_id + "] " + what;
}

// ---------------------------------------------------------------------------
// JobId extraction helper — safe when request or identity absent (AC-3)
// ---------------------------------------------------------------------------
template <typename Request>
inline std::string ExtractJobId(const Request* request) {
    if (!request) return "";
    // Duck-typed: RedactDocumentRequest has has_identity() + identity().job_id()
    // Use SFINAE-free runtime check via has_identity() where available.
    // For generic Request that lacks has_identity, this will fail to compile
    // — caller should pass job_id directly instead.
    if (!request->has_identity()) return "";
    return request->identity().job_id();
}

// String overload for already-extracted job_id (used by catch blocks where
// request may be null — caller does request && request->has_identity() check).
inline std::string ExtractJobIdFromRequest(const std::string& job_id) {
    return job_id;
}

// ---------------------------------------------------------------------------
// Traceability NATS stub — best-effort, never throws (AC-3)
// ---------------------------------------------------------------------------
inline void PublishTraceabilityError(
    const std::string& job_id,
    const std::string& tenant_id,
    const std::string& app_id,
    const std::string& traceparent,
    const std::string& method,
    grpc::StatusCode code,
    const std::string& what) {
    // No NATS client wired in cpp_ml_worker yet — structure is the contract.
    // When a NATS publisher is injected, replace body with:
    //   try { nats_client->Publish("traceability.error", payload); }
    //   catch (...) { spdlog::warn("Traceability NATS publish failed for job {}", job_id); }
    // Swallow is intentional — NATS failure must not change returned Status.
    try {
        (void)job_id;
        (void)tenant_id;
        (void)app_id;
        (void)traceparent;
        (void)method;
        (void)code;
        (void)what;
        // no-op until TraceabilityPublisher injected — one-line injection point
    } catch (...) {
        try {
            spdlog::warn("Traceability NATS publish failed for job {}", job_id.empty() ? "unknown" : job_id);
        } catch (...) {}
    }
}

// ---------------------------------------------------------------------------
// EmitDiagnostics — dual sink, never throws (AC-3)
// ---------------------------------------------------------------------------
inline void EmitDiagnostics(
    const std::string& job_id,
    const std::string& traceparent,
    const std::string& method,
    grpc::StatusCode code,
    const std::string& what,
    const std::string& tenant_id = "",
    const std::string& app_id = "") {
    // Sink (a) stderr via spdlog — exactly one line per failure, with
    // job_id, method, StatusCode, traceparent, what()
    try {
        const char* code_name = (code == grpc::StatusCode::INVALID_ARGUMENT) ? "INVALID_ARGUMENT" : "INTERNAL";
        spdlog::error("[JOB: {}] gRPC {} failed code={} traceparent={} what={}",
            job_id.empty() ? "unknown" : job_id,
            method,
            code_name,
            traceparent.empty() ? "-" : traceparent,
            what);
    } catch (...) {}
    // Sink (b) Traceability NATS — best-effort
    try {
        PublishTraceabilityError(job_id, tenant_id, app_id, traceparent, method, code, what);
    } catch (...) {
        try {
            spdlog::warn("Traceability NATS publish failed for job {}", job_id.empty() ? "unknown" : job_id);
        } catch (...) {}
    }
}

// ---------------------------------------------------------------------------
// Typed GrpcStatusFromException overloads — each embeds job_id and emits
// diagnostics, returns non-OK Status without throwing (AC-1, AC-2)
// ---------------------------------------------------------------------------
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

// Dispatch from current_exception — allowed to rethrow internally because
// it is immediately caught inside the same function (never reaches RPC frame).
// Useful for generic catch-all sites that want a single call.
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

// ---------------------------------------------------------------------------
// WrapGrpc — templated envelope covering every Service:: override (AC-4)
// ---------------------------------------------------------------------------
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
