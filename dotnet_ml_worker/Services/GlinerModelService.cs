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
    public bool IsPii { get; set; }
}

public interface IModelService
{
    IReadOnlyList<PiiToken> Predict(string text);
}

public class GlinerModelService : IModelService, IDisposable
{
    private readonly Tokenizer _tokenizer;
    private readonly InferenceSession _session;

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

        using var stream = File.OpenRead(tokenizerPath);
        _tokenizer = SentencePieceTokenizer.Create(stream, false, false, null);
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
        
        for (int i = 0; i < tokenIds.Length; i++)
        {
            // The model downloaded might not have classes embedded resulting in 0 length logits. Gracefully handle it.
            var isPii = logits.Length > i ? logits.GetValue(i) > 0.5f : false;
            var tokenText = _tokenizer.Decode(new[] { (int)tokenIds[i] });
            resultTokens.Add(new PiiToken { Text = tokenText, IsPii = isPii });
        }

        return resultTokens;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
