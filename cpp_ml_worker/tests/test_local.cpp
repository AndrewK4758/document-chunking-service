#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>

#include "../src/MuPDFWrapper.h"
#include "../src/MLInferenceEngine.h"

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::info);
    if (argc < 2) {
        spdlog::error("Usage: {} <path_to_pdf>", argv[0]);
        return 1;
    }

    std::string pdf_path = argv[1];
    spdlog::info("Loading PDF from: {}", pdf_path);

    // Read the PDF file into a string buffer
    std::ifstream file(pdf_path, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("Error: Could not open file {}", pdf_path);
        return 1;
    }
    
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string pdf_data = ss.str();
    
    spdlog::info("Read {} bytes.", pdf_data.size());

    try {
        spdlog::info("Initializing MuPDFWrapper...");
        MuPDFWrapper mupdf(pdf_data);
        int page_count = mupdf.GetPageCount();
        spdlog::info("PDF has {} pages.", page_count);

        std::string model_path = "/home/ak/projects/ml/model.onnx";
        std::string tokenizer_path = "/home/ak/projects/ml/tokenizer.json";
        
        spdlog::info("Loading ML Engine with model: {}", model_path);
        auto tokenizer = CreateRealTokenizer(tokenizer_path);
        auto onnx_runner = CreateRealOnnxRunner(model_path);
        MLInferenceEngine engine(std::move(tokenizer), std::move(onnx_runner));

        for (int i = 0; i < page_count; i++) {
            spdlog::info("\n--- Processing Page {} ---", i+1);
            auto extracted = mupdf.ExtractPageText(i);
            std::string text = extracted.text;
            auto char_bboxes = extracted.char_bboxes;
            
            spdlog::info("Extracted text length: {} characters.", text.size());
            spdlog::info("Extracted {} character bounding boxes.", char_bboxes.size());
            
            if (text.size() > 100) {
                spdlog::info("Preview: {}...", text.substr(0, 100));
            } else {
                spdlog::info("Preview: {}", text);
            }

            spdlog::info("Running GLiNER Inference...");
            std::vector<fz_rect> redaction_bboxes = engine.CalculateRedactionBBoxes(text, char_bboxes);
            
            spdlog::info("Found {} redaction blocks.", redaction_bboxes.size());
            for (size_t j = 0; j < redaction_bboxes.size(); j++) {
                auto& bbox = redaction_bboxes[j];
                spdlog::info("  - Block {}: ({}, {}) to ({}, {})", j, bbox.x0, bbox.y0, bbox.x1, bbox.y1);
            }
            
            spdlog::info("Applying redactions to PDF...");
            mupdf.ApplyRedactions(i, redaction_bboxes);
        }

        spdlog::info("\nGenerating redacted PDF...");
        std::string redacted_pdf = mupdf.GetRedactedPdf();
        
        std::string out_path = pdf_path + ".redacted.pdf";
        std::ofstream out_file(out_path, std::ios::binary);
        out_file.write(redacted_pdf.data(), redacted_pdf.size());
        
        spdlog::info("Successfully saved redacted PDF to: {}", out_path);
        
    } catch (const std::exception& e) {
        spdlog::error("Caught Exception: {}", e.what());
        return 1;
    }

    return 0;
}
