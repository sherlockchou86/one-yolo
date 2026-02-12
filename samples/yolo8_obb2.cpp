#include "Yolo.h"
using namespace yolo;

int main() {
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "dotav1 obb task using yolov8n(model from ultralytics/ultralytics repo), draw annotated image by myself.";
    cfg.version     = YoloVersion::YOLO8;
    cfg.task        = YoloTaskType::OBB;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/ultralytics/yolov8n-obb.onnx";
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
    result.to_csv();                 // convert structured result to csv and print

    /* 6. draw & show annotated image by myself instead of YoloResult::show() */
    auto rboxes  = result.rboxes();   // get rotated boxes in obb task
    auto cls_ids = result.cls_ids();  // get class ids in obb task
    auto confs   = result.confs();    // get confidences in obb task
    auto labels  = result.labels();   // get labels in obb task

    auto& canvas = image1;            // set original image as canvas
    for (size_t i = 0; i < rboxes.size(); i++) {
        // get 4 points of rotated box
        std::vector<cv::Point2f> vertices_f;
        rboxes[i].points(vertices_f);
        std::vector<cv::Point> i_vertices;
        for (int i = 0; i < 4; i++)
            i_vertices.push_back(cv::Point(cvRound(vertices_f[i].x), cvRound(vertices_f[i].y)));
        // draw rotated box refer to the 4 points
        cv::polylines(
            canvas, i_vertices, 
            true, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        // draw class id, label, confidence near the 2nd point
        cv::putText(
            canvas, 
            std::to_string(cls_ids[i]) + ", " + labels[i] + ", " + std::to_string(confs[i]), 
            i_vertices[1], 1, 1, cv::Scalar(0, 0, 255));
    }
    
    /* show annotated image */
    cv::imshow("annotated image", canvas);
    cv::waitKey(0);
}