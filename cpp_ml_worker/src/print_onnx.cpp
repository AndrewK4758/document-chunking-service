#include <spdlog/spdlog.h>
#include <onnxruntime_cxx_api.h>

int main() {
    spdlog::set_level(spdlog::level::info);
    const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    const Ort::Session session(env, "/home/ak/projects/ml/model.onnx", session_options);

    const size_t num_input_nodes = session.GetInputCount();
    spdlog::info("Number of inputs = {}", num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(i, allocator);
        spdlog::info("Input {} : name={}", i, input_name.get());

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        spdlog::info("   type={}", static_cast<int>(tensor_info.GetElementType()));
        
        auto input_node_dims = tensor_info.GetShape();
        std::string shape_str = "[";
        for (size_t j = 0; j < input_node_dims.size(); j++) {
            shape_str += std::to_string(input_node_dims[j]) + (j < input_node_dims.size() - 1 ? ", " : "");
        }
        shape_str += "]";
        spdlog::info("   shape={}", shape_str);
    }
    return 0;
}
