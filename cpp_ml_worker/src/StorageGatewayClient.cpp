#include "StorageGatewayClient.h"
#include <stdexcept>

using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientWriter;
using grpc::Status;
using shared_protos::DownloadRequest;
using shared_protos::DownloadResponse;
using shared_protos::UploadRequest;
using shared_protos::UploadResponse;
using shared_protos::UploadMetadata;

StorageGatewayClient::StorageGatewayClient(const std::shared_ptr<grpc::Channel> &channel)
    : stub_(shared_protos::StorageGatewayService::NewStub(channel)) {}

std::string StorageGatewayClient::Download(const std::string& uri, const document::models::v1::InfrastructureIdentity& identity) const {
    DownloadRequest request;
    request.set_storage_uri(uri);
    *request.mutable_identity() = identity;

    ClientContext context;
    std::unique_ptr<ClientReader<DownloadResponse>> reader(
        stub_->DownloadDocument(&context, request));

    std::string raw_bytes;
    DownloadResponse response;
    while (reader->Read(&response)) {
        raw_bytes.append(response.chunk_data());
    }

    Status status = reader->Finish();
    if (!status.ok()) {
        throw std::runtime_error("Failed to download document from storage gateway: " + status.error_message());
    }
    if (raw_bytes.empty()) {
        throw std::runtime_error("Downloaded document is empty");
    }
    return raw_bytes;
}

std::string StorageGatewayClient::Upload(const std::string& original_filename, const document::models::v1::InfrastructureIdentity& identity, const std::string& data) const {
    ClientContext context;
    UploadResponse response;
    std::unique_ptr<ClientWriter<UploadRequest>> writer(
        stub_->UploadDocument(&context, &response));

    UploadRequest meta_req;
    UploadMetadata* meta = meta_req.mutable_metadata();
    *meta->mutable_identity() = identity;
    meta->set_original_filename(original_filename);
    if (!writer->Write(meta_req)) {
        throw std::runtime_error("Failed to write metadata to storage gateway");
    }

    UploadRequest data_req;
    data_req.set_chunk_data(data);
    if (!writer->Write(data_req)) {
        throw std::runtime_error("Failed to write chunk data to storage gateway");
    }

    writer->WritesDone();
    Status status = writer->Finish();
    if (!status.ok() || !response.success()) {
        throw std::runtime_error("Failed to upload redacted document to storage gateway: " + status.error_message() + " | " + response.error_message());
    }
    return response.storage_uri();
}
