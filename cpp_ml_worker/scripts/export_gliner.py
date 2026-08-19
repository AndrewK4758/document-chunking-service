import os
import torch
import torch.nn as nn
from gliner import GLiNER

def export_gliner(model_name="urchade/gliner_base", output_dir="/home/ak/projects/ml/"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {model_name}...")
    model = GLiNER.from_pretrained(model_name)
    model.eval()

    class GLiNERExportWrapper(nn.Module):
        def __init__(self, gliner_model):
            super().__init__()
            self.gliner_model = gliner_model

        def forward(self, input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask):
            out = self.gliner_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                words_mask=words_mask,
                text_lengths=text_lengths,
                span_idx=span_idx,
                span_mask=span_mask
            )
            return out.logits

    wrapper = GLiNERExportWrapper(model)
    text = "The borrower is John Reyes and his loan number is 3573409492."
    labels = ["person", "address", "phone number", "email", "loan number"]

    prepared = model.prepare_batch([text], labels)
    collator = model.create_collator()
    batch = model.collate_batch(prepared["input_x"], prepared["entity_types"], collator)

    dummy_inputs = (
        batch["input_ids"],
        batch["attention_mask"],
        batch["words_mask"],
        batch["text_lengths"],
        batch["span_idx"],
        batch["span_mask"]
    )

    input_names = ["input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "words_mask": {0: "batch_size", 1: "seq_len"},
        "text_lengths": {0: "batch_size", 1: "num_texts"},
        "span_idx": {0: "batch_size", 1: "num_spans"},
        "span_mask": {0: "batch_size", 1: "num_spans"},
        "logits": {0: "batch_size", 1: "num_words", 2: "max_span_len", 3: "num_classes"}
    }

    out_onnx = os.path.join(output_dir, "model.onnx")
    print(f"Exporting ONNX to: {out_onnx}")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        out_onnx,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )

    print(f"Saving tokenizer to: {output_dir}")
    model.data_processor.transformer_tokenizer.save_pretrained(output_dir)
    print("Export completed successfully!")

if __name__ == "__main__":
    export_gliner()
