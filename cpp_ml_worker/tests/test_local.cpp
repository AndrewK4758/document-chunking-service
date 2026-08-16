#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>

#include "../src/MuPDFWrapper.h"
#include "../src/MLInferenceEngine.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_pdf>" << std::endl;
        return 1;
    }

    std::string pdf_path = argv[1];
    std::cout << "Loading PDF from: " << pdf_path << std::endl;

    // Read the PDF file into a string buffer
    std::ifstream file(pdf_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << pdf_path << std::endl;
        return 1;
    }
    
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string pdf_data = ss.str();
    
    std::cout << "Read " << pdf_data.size() << " bytes." << std::endl;

    try {
        std::cout << "Initializing MuPDFWrapper..." << std::endl;
        MuPDFWrapper mupdf(pdf_data);
        int page_count = mupdf.GetPageCount();
        std::cout << "PDF has " << page_count << " pages." << std::endl;

        std::string model_path = "/home/ak/projects/ml/model.onnx";
        std::string tokenizer_path = "/home/ak/projects/ml/tokenizer.json";
        
        std::cout << "Loading ML Engine with model: " << model_path << std::endl;
        auto tokenizer = CreateRealTokenizer(tokenizer_path);
        auto onnx_runner = CreateRealOnnxRunner(model_path);
        MLInferenceEngine engine(std::move(tokenizer), std::move(onnx_runner));

        for (int i = 0; i < page_count; i++) {
            std::cout << "\n--- Processing Page " << (i+1) << " ---" << std::endl;
            auto extracted = mupdf.ExtractPageText(i);
            std::string text = extracted.text;
            auto char_bboxes = extracted.char_bboxes;
            
            std::cout << "Extracted text length: " << text.size() << " characters." << std::endl;
            std::cout << "Extracted " << char_bboxes.size() << " character bounding boxes." << std::endl;
            
            if (text.size() > 100) {
                std::cout << "Preview: " << text.substr(0, 100) << "..." << std::endl;
            } else {
                std::cout << "Preview: " << text << std::endl;
            }

            std::cout << "Running GLiNER Inference..." << std::endl;
            std::vector<fz_rect> redaction_bboxes = engine.CalculateRedactionBBoxes(text, char_bboxes);
            
            std::cout << "Found " << redaction_bboxes.size() << " redaction blocks." << std::endl;
            for (size_t j = 0; j < redaction_bboxes.size(); j++) {
                auto& bbox = redaction_bboxes[j];
                std::cout << "  - Block " << j << ": (" << bbox.x0 << ", " << bbox.y0 << ") to (" << bbox.x1 << ", " << bbox.y1 << ")" << std::endl;
            }
            
            std::cout << "Applying redactions to PDF..." << std::endl;
            mupdf.ApplyRedactions(i, redaction_bboxes);
        }

        std::cout << "\nGenerating redacted PDF..." << std::endl;
        std::string redacted_pdf = mupdf.GetRedactedPdf();
        
        std::string out_path = pdf_path + ".redacted.pdf";
        std::ofstream out_file(out_path, std::ios::binary);
        out_file.write(redacted_pdf.data(), redacted_pdf.size());
        
        std::cout << "Successfully saved redacted PDF to: " << out_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Caught Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
