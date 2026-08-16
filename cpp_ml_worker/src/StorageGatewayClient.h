#pragma once

#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include "storage_gateway.grpc.pb.h"
#include "models.pb.h"

class StorageGatewayClient {
public:
    explicit StorageGatewayClient(const std::shared_ptr<grpc::Channel> &channel);

    [[nodiscard]] std::string Download(const std::string& uri, const document::models::v1::InfrastructureIdentity& identity) const;
    [[nodiscard]] std::string Upload(const std::string& original_filename, const document::models::v1::InfrastructureIdentity& identity, const std::string& data) const;

private:
    std::unique_ptr<shared_protos::StorageGatewayService::Stub> stub_;
};
