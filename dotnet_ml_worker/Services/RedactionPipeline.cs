using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Ml.Worker.V1;
using PdfSharpCore.Drawing;
using PdfSharpCore.Pdf;
using PdfSharpCore.Pdf.IO;
using UglyToad.PdfPig;

namespace dotnet_ml_worker.Services;

public interface IRedactionPipeline
{
    (byte[] RedactedBytes, int RedactedCount) Redact(byte[] inputPdfBytes, IReadOnlyList<PiiType> targetTypes);
}

public class RedactionPipeline : IRedactionPipeline
{
    public (byte[] RedactedBytes, int RedactedCount) Redact(byte[] inputPdfBytes, IReadOnlyList<PiiType> targetTypes)
    {
        if (inputPdfBytes == null || inputPdfBytes.Length == 0)
        {
            throw new ArgumentException("Input PDF bytes cannot be null or empty", nameof(inputPdfBytes));
        }

        var targets = targetTypes.ToHashSet();
        var redactions = new List<RedactionBox>();

        // Step 1: Read with PdfPig and extract bounding boxes
        using (var pigDoc = UglyToad.PdfPig.PdfDocument.Open(inputPdfBytes))
        {
            for (int i = 1; i <= pigDoc.NumberOfPages; i++)
            {
                var page = pigDoc.GetPage(i);
                foreach (var word in page.GetWords())
                {
                    var piiType = MockInference(word.Text);
                    if (piiType != PiiType.Unspecified && targets.Contains(piiType))
                    {
                        redactions.Add(new RedactionBox(
                            i,
                            word.BoundingBox.Left,
                            word.BoundingBox.Bottom,
                            word.BoundingBox.Width,
                            word.BoundingBox.Height
                        ));
                    }
                }
            }
        }

        // Step 2: Draw black rectangles using PdfSharpCore
        int redactedCount = redactions.Count;
        if (redactedCount == 0)
        {
            return (inputPdfBytes, 0);
        }

        using (var memoryStream = new MemoryStream(inputPdfBytes))
        using (var pdfSharpDoc = PdfReader.Open(memoryStream, PdfDocumentOpenMode.Modify))
        {
            foreach (var group in redactions.GroupBy(r => r.PageNumber))
            {
                var page = pdfSharpDoc.Pages[group.Key - 1]; // PdfSharp pages are 0-indexed
                using (var gfx = XGraphics.FromPdfPage(page))
                {
                    foreach (var rect in group)
                    {
                        // PdfPig's origin is Bottom-Left.
                        // PdfSharpCore's origin is Top-Left.
                        var top = page.Height.Point - rect.Bottom - rect.Height;

                        var xRect = new XRect(rect.Left, top, rect.Width, rect.Height);
                        gfx.DrawRectangle(XBrushes.Black, xRect);
                    }
                }
            }

            using (var outStream = new MemoryStream())
            {
                pdfSharpDoc.Save(outStream);
                return (outStream.ToArray(), redactedCount);
            }
        }
    }

    private static PiiType MockInference(string text)
    {
        var lower = text.ToLowerInvariant();
        if (lower.Contains("ssn") || lower.Contains("000-00-0000"))
        {
            return PiiType.Ssn;
        }
        if (lower == "john" || lower == "doe")
        {
            return PiiType.PersonName;
        }
        if (lower.Contains("@") && lower.Contains(".com"))
        {
            return PiiType.Email;
        }
        if (lower.StartsWith("$"))
        {
            return PiiType.Money;
        }

        return PiiType.Unspecified;
    }

    private readonly record struct RedactionBox(int PageNumber, double Left, double Bottom, double Width, double Height);
}
