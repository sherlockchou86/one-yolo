#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "dotav1 obb task using yolov11n(model from ultralytics/ultralytics repo), get structured results separately.";
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
    auto image1 = cv::imread("./vp_data/test_images/obb/0.png");

    // 4. run predict
    auto result = model(image1);

    /* 5. show and print */
    result.info();                   // print summary 
    result.to_json(true, false);     // convert structured result to json(no indent) and print
    result.to_csv();                 // convert structured result to csv
    result.show();                   // show annotated image with block mode, press any key to exit program

    /* 6. get structured results separately */
    auto rboxes  = result.rboxes();   // get rotated boxes in obb task
    auto cls_ids = result.cls_ids();  // get class ids in obb task
    auto confs   = result.confs();    // get confidences in obb task
    auto labels  = result.labels();   // get labels in obb task  

    /* 7. print structured results separately (just like in python) */
    std::cout << "rotated boxes of detected objects:" << std::endl;
    std::cout << rboxes << std::endl;
    // print rotated boxes of 2 objects:
    // [{"angle":69.96593475341797,"cx":901.0,"cy":694.0,"height":56.0,"width":38.0},{"angle":19.453123092651367,"cx":341.0,"cy":697.0,"height":47.0,"width":22.0}]
    // you can also use: 
    // std::cout << yolo::to_string(rboxes)     << std::endl;    

    std::cout << "class ids of detected objects:" << std::endl;
    std::cout << cls_ids << std::endl;
    // print class ids of 2 objects:
    // [14,4]
    // you can also use: 
    // std::cout << yolo::to_string(cls_ids)     << std::endl;    

    std::cout << "confidences of detected objects:" << std::endl;
    std::cout << confs << std::endl;
    // print confidences of 2 objects:
    // [0.8414784669876099,0.8777657151222229]
    // you can also use: 
    // std::cout << yolo::to_string(confs)     << std::endl;    

    std::cout << "labels of detected objects:" << std::endl;
    std::cout << labels << std::endl;
    // print labels of 2 objects:
    // ["swimming pool","tennis court"]
    // you can also use: 
    // std::cout << yolo::to_string(labels)     << std::endl;    
}