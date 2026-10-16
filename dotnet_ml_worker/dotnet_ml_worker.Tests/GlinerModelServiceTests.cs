using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using dotnet_ml_worker.Services;
using FluentAssertions;
using Xunit;

namespace dotnet_ml_worker.Tests;

public class GlinerModelServiceTests
{
    private readonly string _validModelDir = "/home/ak/projects/ml";

    [Fact]
    public void Constructor_ModelFilesExist_ShouldInitializeSuccessfully()
    {
        var service = new GlinerModelService(_validModelDir);

        service.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_ModelFilesMissing_ShouldThrowException()
    {
        var invalidDir = "/path/to/nowhere";

        Action act = () => new GlinerModelService(invalidDir);

        act.Should().Throw<Exception>();
    }

    [Fact]
    public void Predict_MalformedModel_ShouldThrowException()
    {
        var service = new GlinerModelService(_validModelDir);
        var text = "John Doe lives in New York and his email is john.doe@example.com.";

        Action act = () => service.Predict(text);

        act.Should().Throw<InvalidOperationException>()
           .WithMessage("*malformed logits*");
    }
}
