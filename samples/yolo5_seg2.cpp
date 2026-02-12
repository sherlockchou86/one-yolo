#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "lane segmentation task using yolov5s(custom model), just show mask area.";
    cfg.version     = YoloVersion::YOLO5;
    cfg.task        = YoloTaskType::SEG;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/lane/das_yolov5s-seg_c2_b1_20260205.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 384;
    cfg.iou_thresh  = 0.65f;  // for denser targets
    cfg.approx_f    = 0.003f;
    cfg.batch_size  = 1;
    cfg.num_classes = 2;
    cfg.names       = {"normal", "special"};

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* open video and predict frames in a loop */
    cv::VideoCapture cap("./vp_data/test_video/vehicle_count.mp4");
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

        // construct draw parameter
        DrawParam param;
        param.boxes = false;         // do not show boxes(and class ids / confidences / labels)
        param.mask_line_width = 2;   // bold
        if (results[0].show(
            false, 0.5f, param,      // show annotated image & input image(640*384) with unblock mode
            false, true) == 27) {    // exit loop if user has pressed ESC
            break;
        }
    }
}