#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "imagenet classification task using yolov8n(model from ultralytics/ultralytics repo)";
    cfg.version     = YoloVersion::YOLO8;
    cfg.task        = YoloTaskType::CLS;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/ultralytics/yolov8n-cls_b2.onnx";
    cfg.input_w     = 224;
    cfg.input_h     = 224;
    cfg.batch_size  = 0;    // dynamic batch
    cfg.num_classes = 1000;
    cfg.names       = IMAGENET_NAMES;

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. collect test images */
    auto image0 = cv::imread("./vp_data/test_images/vehicle_cls/1.jpg");
    auto image1 = cv::imread("./vp_data/test_images/vehicle_cls/2.jpg");
    auto image2 = cv::imread("./vp_data/test_images/vehicle_cls/3.jpg");

    /* 
     * 4. predict with single image and get YoloResult
    */
    auto result = model(image1);
    result.info();                    // print summary
    result.to_json(true);             // convert structured result to json and print
    result.to_csv(true);              // convert structured result to csv and print
    result.show();                    // show annotated image with block mode(press any key to continue)

    /* 
     * 5. predict with multi images and get YoloResults
    */
    auto results = model(std::vector<cv::Mat>({image0, image1, image2}));
    results[0].info();                    // print summary
    results[0].to_json(true);             // convert structured result to json and print
    results[0].to_csv(true);              // convert structured result to csv and print
    results[0].show(false);               // show annotated image with **unblock** mode

    results[1].info();                    // print summary
    results[1].to_json(true);             // convert structured result to json and print
    results[1].to_csv(true);              // convert structured result to csv and print
    results[1].show(false);               // show annotated image with **unblock** mode

    results[2].info();                    // print summary
    results[2].to_json(true);             // convert structured result to json and print
    results[2].to_csv(true);              // convert structured result to csv and print
    results[2].show();                    // show annotated image with block mode, press any key to exit program


    /**
     * you can also get structured results like below:
     * auto top1       = result.top1();            // get top1 class id from classification task
     * auto top1_conf  = result.top1_conf();       // get top1 confidence from classification task
     * auto top1_label = result.top1_label();      // get top1 label from classification task
     * 
     * auto top5        = result.top5();           // get top5 class ids from classification task
     * auto top5_confs  = result.top5_confs();     // get top5 confidences from classification task
     * auto top5_labels = result.top5_labels();    // get top5 labels from classification task
     *  
    */
}