#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "weather classification task using yolov5s(custom model)";
    cfg.version     = YoloVersion::YOLO5;
    cfg.task        = YoloTaskType::CLS;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/det_cls/weather_yolov5s-cls_c4_b1_20250807.onnx";
    cfg.input_w     = 224;
    cfg.input_h     = 224;
    cfg.batch_size  = 1;
    cfg.mean        = {0.485, 0.456, 0.406};
    cfg.std         = {0.229, 0.224, 0.225};
    cfg.num_classes = 4;
    cfg.names       = {"fog", "normal", "rain", "snow"};

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. collect test images */
    auto image0 = cv::imread("./vp_data/test_images/vehicle_cls/20.jpg");
    auto image1 = cv::imread("./vp_data/test_images/vehicle_cls/21.jpg");
    auto image2 = cv::imread("./vp_data/test_images/vehicle_cls/22.jpg");

    /* 4. predict one by one and get YoloResult,
          you can also predict with batch mode and get YoloResults.
    */
    auto result = model.predict(image0);
    result.info();                    // print summary
    result.to_json(true);             // convert structured result to json and print
    result.to_csv(true);              // convert structured result to csv and print
    result.show();                    // show annotated image with block mode(press any key to continue)

    result = model.predict(image1);
    result.info();                    // print summary
    result.to_json(true);             // convert structured result to json and print
    result.to_csv(true);              // convert structured result to csv and print
    result.show();                    // show annotated image with block mode(press any key to continue)

    result = model.predict(image2);
    result.info();                    // print summary
    result.to_json(true);             // convert structured result to json and print
    result.to_csv(true);              // convert structured result to csv and print
    result.show();                    // show annotated image with block mode, press any key to exit program
}