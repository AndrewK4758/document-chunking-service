using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using Ml.Worker.V1;

namespace dotnet_ml_worker.Services;

public class PiiToken
{
    public string Text { get; set; } = string.Empty;
    public PiiType Type { get; set; }
}

public interface IModelService
{
    IReadOnlyList<PiiToken> Predict(string text);
}

public class GlinerModelService : IModelService, IDisposable
{
    private readonly Tokenizer _tokenizer;
    private readonly InferenceSession _session;
    private readonly Dictionary<int, string> _id2label = new();

    public GlinerModelService(string modelDir)
    {
        var modelPath = Path.Combine(modelDir, "model.onnx");
        var tokenizerPath = Path.Combine(modelDir, "spm.model"); // Use spm.model for SentencePiece

        if (!File.Exists(modelPath))
        {
            throw new Exception($"Model file missing at {modelPath}");
        }
        if (!File.Exists(tokenizerPath))
        {
            throw new Exception($"Tokenizer file missing at {tokenizerPath}");
        }

        _session = new InferenceSession(modelPath);

        if (_session.ModelMetadata.CustomMetadataMap.TryGetValue("id2label", out var id2labelStr))
        {
            var parsed = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, string>>(id2labelStr);
            if (parsed != null)
            {
                foreach (var kvp in parsed)
                {
                    if (int.TryParse(kvp.Key, out int id))
                    {
                        _id2label[id] = kvp.Value;
                    }
                }
            }
        }

        using var stream = File.OpenRead(tokenizerPath);
        _tokenizer = SentencePieceTokenizer.Create(stream, false, false, null);
    }

    private PiiType MapToPiiType(string label)
    {
        return label.ToUpperInvariant() switch
        {
            "PERSON" => PiiType.PersonName,
            "SSN" => PiiType.Ssn,
            "ORG" => PiiType.Unspecified,
            "LOC" => PiiType.Address,
            "ADDRESS" => PiiType.Address,
            "PHONE" => PiiType.PhoneNumber,
            "EMAIL" => PiiType.Email,
            "MONEY" => PiiType.Money,
            "PERCENTAGE" => PiiType.Percentage,
            "BARCODE" => PiiType.Barcode,
            "QRCODE" => PiiType.QrCode,
            "ESIGNATURE" => PiiType.Esignature,
            "IMAGE" => PiiType.Image,
            "TIME" => PiiType.Time,
            "LENDER" => PiiType.Lender,
            "ORDERNUMBER" => PiiType.OrderNumber,
            "LOANNUMBER" => PiiType.LoanNumber,
            "AGENT" => PiiType.Agent,
            "SIGNATURE" => PiiType.Signature,
            _ => PiiType.Unspecified
        };
    }

    public IReadOnlyList<PiiToken> Predict(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<PiiToken>();

        var tokenIds = _tokenizer.EncodeToIds(text).Select(x => (long)x).ToArray();
        if (tokenIds.Length == 0) return Array.Empty<PiiToken>();

        int maxLen = 12;
        int numSpans = tokenIds.Length * maxLen;

        var tensor = new DenseTensor<long>(tokenIds, new[] { 1, tokenIds.Length });
        var attentionMask = new DenseTensor<long>(Enumerable.Repeat(1L, tokenIds.Length).ToArray(), new[] { 1, tokenIds.Length });
        var wordsMask = new DenseTensor<long>(Enumerable.Repeat(1L, tokenIds.Length).ToArray(), new[] { 1, tokenIds.Length });
        var textLengths = new DenseTensor<long>(new[] { (long)tokenIds.Length }, new[] { 1, 1 });

        var spanIdxArr = new long[numSpans * 2];
        var spanMaskArr = new bool[numSpans];
        int idx = 0;
        for (int i = 0; i < tokenIds.Length; i++)
        {
            for (int j = 0; j < maxLen; j++)
            {
                int end = i + j;
                if (end < tokenIds.Length)
                {
                    spanIdxArr[idx * 2] = i;
                    spanIdxArr[idx * 2 + 1] = end;
                    spanMaskArr[idx] = true;
                }
                idx++;
            }
        }
        var spanIdx = new DenseTensor<long>(spanIdxArr, new[] { 1, numSpans, 2 });
        var spanMask = new DenseTensor<bool>(spanMaskArr, new[] { 1, numSpans });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", tensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
            NamedOnnxValue.CreateFromTensor("words_mask", wordsMask),
            NamedOnnxValue.CreateFromTensor("text_lengths", textLengths),
            NamedOnnxValue.CreateFromTensor("span_idx", spanIdx),
            NamedOnnxValue.CreateFromTensor("span_mask", spanMask)
        };

        using var results = _session.Run(inputs);
        var logits = results.First(v => v.Name == "logits").AsTensor<float>();

        var resultTokens = new List<PiiToken>(tokenIds.Length);
        
        int numClasses = logits.Length > 0 && numSpans > 0 ? (int)(logits.Length / numSpans) : 1;

        for (int i = 0; i < tokenIds.Length; i++)
        {
            if (logits.Length <= i)
            {
                throw new InvalidOperationException($"Model returned malformed logits. Expected at least {i + 1} logits but got {logits.Length}.");
            }

            int bestClassIdx = 0;
            float maxProb = -1f;

            if (numClasses > 1)
            {
                for (int c = 0; c < numClasses; c++)
                {
                    int spanIndex = i * maxLen;
                    int logitIndex = spanIndex * numClasses + c;
                    if (logitIndex < logits.Length)
                    {
                        float val = logits.GetValue(logitIndex);
                        if (val > maxProb)
                        {
                            maxProb = val;
                            bestClassIdx = c;
                        }
                    }
                }
            }
            else
            {
                maxProb = logits.GetValue(i);
            }

            string labelName = _id2label.TryGetValue(bestClassIdx, out var name) ? name : "UNKNOWN";
            var tokenType = MapToPiiType(labelName);

            if (numClasses <= 1)
            {
                tokenType = maxProb > 0.5f ? PiiType.PersonName : PiiType.Unspecified;
            }
            else if (maxProb <= 0.5f)
            {
                tokenType = PiiType.Unspecified;
            }

            var tokenText = _tokenizer.Decode(new[] { (int)tokenIds[i] });
            resultTokens.Add(new PiiToken { Text = tokenText, Type = tokenType });
        }

        return resultTokens;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
