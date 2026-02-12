#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "dotav1 obb task using yolov11n(model from ultralytics/ultralytics repo)";
    cfg.version     = YoloVersion::YOLO11;
    cfg.task        = YoloTaskType::OBB;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/ultralytics/yolo11n-obb.onnx";
    cfg.input_w     = 1024;
    cfg.input_h     = 1024;
    cfg.batch_size  = 1;
    cfg.num_classes = 15;
    cfg.names       = DOTAV1_NAMES;

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. collect test image */
    auto image1 = cv::imread("./vp_data/test_images/obb/1.png");

    // 4. run predict
    auto result = model(image1);

    /* 5. show and print */
    result.info();                   // print summary 
    result.to_json(true, false);     // convert structured result to json(no indent) and print
    result.to_csv();                 // convert structured result to csv
    result.show();                   // show annotated image with block mode, press any key to exit program

    /* 
     * you can also get structured results like below:
     * auto rboxes  = result.rboxes();   // get rotated boxes in obb task
     * auto cls_ids = result.cls_ids();  // get class ids in obb task
     * auto confs   = result.confs();    // get confidences in obb task
     * auto labels  = result.labels();   // get labels in obb task
    */
}