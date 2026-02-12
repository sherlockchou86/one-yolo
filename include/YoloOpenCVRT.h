#include "YoloRuntime.h"


namespace yolo {
    class YoloOpenCVRT: public YoloRuntime
    {
    private:
        cv::dnn::Net __net;
    public:
        YoloOpenCVRT(const std::string& model_path, bool use_cuda = true);
        ~YoloOpenCVRT();
        virtual std::vector<cv::Mat> inference(const cv::Mat& blob) override;
    };
}