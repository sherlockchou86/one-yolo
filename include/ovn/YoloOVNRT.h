#include <openvino/openvino.hpp>
#include "YoloRuntime.h"


namespace yolo {
    /**
     * @brief
     * Yolo runtime based on openvino library.
    */
    class YoloOVNRT: public YoloRuntime
    {
    private:
        ov::Core __core;
        ov::CompiledModel __compiled_model;
    public:
        YoloOVNRT(
            const std::string& model_path, 
            const std::string& device = "CPU");
        ~YoloOVNRT();
        virtual std::vector<cv::Mat> inference(const cv::Mat& blob) override;
    };
}