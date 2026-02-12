#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "imagenet classification task using yolov11n(model from ultralytics/ultralytics repo)";
    cfg.version     = YoloVersion::YOLO11;
    cfg.task        = YoloTaskType::CLS;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/ultralytics/yolo11n-cls_b2.onnx";
    cfg.input_w     = 224;
    cfg.input_h     = 224;
    cfg.batch_size  = 2;
    cfg.num_classes = 1000;
    cfg.names       = IMAGENET_NAMES;

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. collect test images */
    auto image0 = cv::imread("./vp_data/test_images/vehicle_cls/3.jpg");
    auto image1 = cv::imread("./vp_data/test_images/vehicle_cls/40.jpg");

    /* 
     * 4. predict with images and get YoloResults
    */
    auto results = model(std::vector<cv::Mat>{image0, image1});
    results[0].info();                    // print summary
    results[0].to_json(true);             // convert structured result to json and print
    results[0].to_csv(true);              // convert structured result to csv and print
    results[0].show();                    // show annotated image with block mode(press any key to continue)

    /**
     * you can also get structured results like below:
     * auto top1       = results[0].top1();            // get top1 class id from classification task
     * auto top1_conf  = results[0].top1_conf();       // get top1 confidence from classification task
     * auto top1_label = results[0].top1_label();      // get top1 label from classification task
     * 
     * auto top5        = results[0].top5();           // get top5 class ids from classification task
     * auto top5_confs  = results[0].top5_confs();     // get top5 confidences from classification task
     * auto top5_labels = results[0].top5_labels();    // get top5 labels from classification task
     *  
    */
}