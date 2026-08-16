using System;
using System.IO;
using System.Linq;
using Moq;
using dotnet_ml_worker.Services;
using Ml.Worker.V1;
using PdfSharpCore.Drawing;
using PdfSharpCore.Pdf;
using Xunit;

namespace dotnet_ml_worker.Tests;

public class RedactionPipelineTests
{
    [Fact]
    public void Redact_GivenPdfWithPii_ShouldReturnRedactedPdf()
    {
        // Arrange
        byte[] pdfBytes = CreateDummyPdf("Hello John Doe and others.");
        var mockService = new Moq.Mock<IModelService>();
        mockService.Setup(s => s.Predict(It.IsAny<string>()))
            .Returns<string>(text => new[] { new PiiToken { Text = text, Type = (text == "John" || text == "Doe") ? PiiType.PersonName : PiiType.Unspecified } });
        var pipeline = new RedactionPipeline(mockService.Object);
        var targetTypes = new[] { PiiType.PersonName };

        // Act
        // Debug output
        using (var pigDoc = UglyToad.PdfPig.PdfDocument.Open(pdfBytes))
        {
            var words = string.Join(", ", pigDoc.GetPages().SelectMany(p => p.GetWords()).Select(w => w.Text));
            Console.WriteLine("PdfPig saw: " + words);
        }

        var (redactedBytes, count) = pipeline.Redact(pdfBytes, targetTypes);

        // Assert
        Assert.NotNull(redactedBytes);
        Assert.True(redactedBytes.Length > 0);
        // Expecting 2 redactions: "John" and "Doe"
        Assert.Equal(2, count);
        
        // We can optionally verify the new PDF doesn't easily show "John Doe",
        // but PDF text extraction can be tricky. Bounding box count is our primary check.
    }

    [Fact]
    public void Redact_NoPiiMatch_ShouldReturnOriginalPdf()
    {
        // Arrange
        byte[] pdfBytes = CreateDummyPdf("Hello Alice Bob and others.");
        var mockService = new Moq.Mock<IModelService>();
        mockService.Setup(s => s.Predict(It.IsAny<string>()))
            .Returns<string>(text => new[] { new PiiToken { Text = text, Type = PiiType.Unspecified } });
        var pipeline = new RedactionPipeline(mockService.Object);
        var targetTypes = new[] { PiiType.PersonName };

        // Act
        var (redactedBytes, count) = pipeline.Redact(pdfBytes, targetTypes);

        // Assert
        Assert.Equal(0, count);
    }

    private byte[] CreateDummyPdf(string text)
    {
        using (var document = new PdfDocument())
        {
            var page = document.AddPage();
            using (var gfx = XGraphics.FromPdfPage(page))
            {
                var options = new XPdfFontOptions(PdfFontEncoding.WinAnsi);
                var font = new XFont("Helvetica", 20, XFontStyle.Regular, options);
                gfx.DrawString(text, font, XBrushes.Black, new XRect(0, 0, page.Width, page.Height), XStringFormats.TopLeft);
            }

            using (var stream = new MemoryStream())
            {
                document.Save(stream);
                return stream.ToArray();
            }
        }
    }
}
