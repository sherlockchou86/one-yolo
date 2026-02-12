#include <onnxruntime_cxx_api.h>
#include "YoloRuntime.h"


namespace yolo {
    /**
     * @brief
     * Yolo runtime based on onnxruntime library.
    */
    class YoloONNXRT: public YoloRuntime
    {
    private:
        Ort::Env __env;
        Ort::Session __session {nullptr};
        Ort::SessionOptions __session_options;
        void ort_forward(
            const cv::Mat& input_4d,
            std::vector<cv::Mat>& outputs
        );
    public:
        YoloONNXRT(
            const std::string& model_path, 
            bool use_cuda = true);
        ~YoloONNXRT();
        virtual std::vector<cv::Mat> inference(const cv::Mat& blob) override;
    };
}