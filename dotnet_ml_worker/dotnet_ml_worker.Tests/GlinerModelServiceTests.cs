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
        // Arrange
        // Act
        var service = new GlinerModelService(_validModelDir);

        // Assert
        service.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_ModelFilesMissing_ShouldThrowException()
    {
        // Arrange
        var invalidDir = "/path/to/nowhere";

        // Act
        Action act = () => new GlinerModelService(invalidDir);

        // Assert
        act.Should().Throw<Exception>();
    }

    [Fact]
    public void Predict_MalformedModel_ShouldThrowException()
    {
        // Arrange
        var service = new GlinerModelService(_validModelDir);
        var text = "John Doe lives in New York and his email is john.doe@example.com.";

        // Act
        Action act = () => service.Predict(text);

        // Assert
        act.Should().Throw<InvalidOperationException>()
           .WithMessage("*malformed logits*");
    }
}
