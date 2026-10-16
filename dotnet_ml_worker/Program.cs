using dotnet_ml_worker.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddGrpc();
builder.Services.AddSingleton<IModelService>(sp => new GlinerModelService("/home/ak/projects/ml"));
builder.Services.AddSingleton<IRedactionPipeline, RedactionPipeline>();

var app = builder.Build();

app.MapGrpcService<MlWorkerServiceImpl>();
app.MapGet("/", () => "Communication with gRPC endpoints must be made through a gRPC client. To learn how to create a client, visit: https://go.microsoft.com/fwlink/?linkid=2086909");

app.Run();
