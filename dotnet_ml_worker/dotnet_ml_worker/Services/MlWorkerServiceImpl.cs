using System;
using System.Threading.Tasks;
using Grpc.Core;
using Ml.Worker.V1;
using Google.Protobuf;

namespace dotnet_ml_worker.Services;

public class MlWorkerServiceImpl : MlWorkerService.MlWorkerServiceBase
{
    private readonly IRedactionPipeline _redactionPipeline;

    public MlWorkerServiceImpl(IRedactionPipeline redactionPipeline)
    {
        _redactionPipeline = redactionPipeline;
    }

    public override Task<PiiRedactionResponse> RedactPii(PiiRedactionRequest request, ServerCallContext context)
    {
        try
        {
            if (request.Identity == null)
            {
                throw new RpcException(new Status(StatusCode.InvalidArgument, "Identity must be provided."));
            }

            if (request.RawFileBytes == null || request.RawFileBytes.IsEmpty)
            {
                throw new RpcException(new Status(StatusCode.InvalidArgument, "RawFileBytes must be provided and non-empty."));
            }

            var (redactedBytes, count) = _redactionPipeline.Redact(request.RawFileBytes.ToByteArray(), request.TargetPiiTypes);

            return Task.FromResult(new PiiRedactionResponse
            {
                Success = true,
                RedactedFileBytes = ByteString.CopyFrom(redactedBytes),
                RedactedBoxesCount = count
            });
        }
        catch (Exception ex)
        {
            return Task.FromResult(new PiiRedactionResponse
            {
                Success = false,
                ErrorMessage = ex.Message
            });
        }
    }
}
