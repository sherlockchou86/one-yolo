#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "video quality classification task using yolov8s(custom model)";
    cfg.version     = YoloVersion::YOLO8;
    cfg.task        = YoloTaskType::CLS;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/det_cls/vqd_v8s_c2_with_preprocess_640_384_b1_20260130.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 384;
    cfg.batch_size  = 0;
    cfg.nchw        = false;    // channel last
    cfg.scale_f     = 1.0f;     // no normalize
    cfg.num_classes = 2;
    cfg.names       = {"normal", "unnormal"};

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. open video and predict frames in a loop */
    cv::VideoCapture cap("./vp_data/test_video/smoke.mp4");
    while (cap.isOpened()) {
        // collect frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        // predict with batch mode (batch size == 1)
        auto results = model(std::vector<cv::Mat>{frame});
        
        // show and print
        results[0].info();           // print summary
        results[0].to_json(true);    // convert structured result to json and print
        results[0].to_csv(true);     // convert structured result to csv and print
        if (results[0].show(
            false, 1.0f, DrawParam(), // show annotated image & input image(640*384) with unblock mode
            false, true) == 27) {     // exit loop if user has pressed ESC
            break;
        }
    }
}