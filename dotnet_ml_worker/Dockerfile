FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /app

COPY --from=shared_protos . /app/shared_protos
COPY . /app/document_processing_service/dotnet_ml_worker

WORKDIR /app/document_processing_service/dotnet_ml_worker
RUN dotnet restore dotnet_ml_worker/dotnet_ml_worker.csproj
RUN dotnet publish dotnet_ml_worker/dotnet_ml_worker.csproj -c Release -o /out

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /out .

EXPOSE 50051

ENTRYPOINT ["dotnet", "dotnet_ml_worker.dll"]
