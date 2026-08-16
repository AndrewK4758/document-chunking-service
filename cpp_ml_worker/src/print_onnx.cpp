#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    const Ort::Session session(env, "/home/ak/projects/ml/model.onnx", session_options);

    const size_t num_input_nodes = session.GetInputCount();
    std::cout << "Number of inputs = " << num_input_nodes << std::endl;

    for (size_t i = 0; i < num_input_nodes; i++) {
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << " : name=" << input_name.get() << std::endl;

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << "   type=" << tensor_info.GetElementType() << std::endl;
        
        auto input_node_dims = tensor_info.GetShape();
        std::cout << "   shape=[";
        for (size_t j = 0; j < input_node_dims.size(); j++) {
            std::cout << input_node_dims[j] << (j < input_node_dims.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }
    return 0;
}
