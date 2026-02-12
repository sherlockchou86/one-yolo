#include "Yolo.h"
#include "track/YoloTracker.h"
#include <chrono>
#include <thread>
using namespace yolo;

int main() {
    // make sure videoio module enabled in OpenCV
    std::cout << cv::getBuildInformation() << std::endl;
    /* 1. construct YoloConfig */
    YoloConfig cfg;
    cfg.desc        = "vehicle detection task using yolov5s(custom model), push annotated image to mp4 file.";
    cfg.version     = YoloVersion::YOLO5;
    cfg.task        = YoloTaskType::DET;
    cfg.target_rt   = YoloTargetRT::OPENCV_CUDA;
    cfg.model_path  = "./vp_data/models/det_cls/vehicle_yolov5s-det_c5_b1_20260129.onnx";
    cfg.input_w     = 640;
    cfg.input_h     = 384;
    cfg.batch_size  = 1;
    cfg.num_classes = 5;
    cfg.names       = {"person", "car", "bus", "truck", "2wheel"};

    /* 2. create Yolo using YoloConfig */
    auto model = Yolo(cfg);
    model.info();

    /* 3. construct YoloTrackConfig */
    YoloTrackConfig t_cfg;
    t_cfg.iou_thresh = 0.6f;

    /* 4. create YoloTracker using YoloTrackConfig */
    auto tracker = YoloTracker(t_cfg);
    tracker.info();

    /* 5. open video and predict frames in a loop */
    /*    push annotated image to mp4 file */
    cv::VideoCapture cap("./vp_data/test_video/rgb.mp4");
    cv::VideoWriter wrt;
    int out_fps;
    cv::Size out_size;
    while (cap.isOpened()) {
        auto start = std::chrono::system_clock::now();
        // collect frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        if (!wrt.isOpened()) {
            out_fps = cap.get(cv::CAP_PROP_FPS);
            out_size = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            if (!wrt.open(
                "./yolo5_det3.mp4",
                cv::CAP_FFMPEG,
                cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 
                out_fps, out_size, true)) {
                std::cout << "failed to open mp4 file..." << std::endl;
                continue;
            }
        }
        
        // predict
        auto result = model(frame);

        // track result
        tracker(result);
        
        // show and print
        result.info();           // print summary
        result.to_json(true);    // convert structured result to json and print
        result.to_csv(true);     // convert structured result to csv and print
        if (result.show(
            false, 1.0f, DrawParam(), // show annotated image & input image(640*384) with unblock mode
            false, true) == 27) {     // exit loop if user has pressed ESC
            break;
        }

        // get annotated image & write to mp4 file
        auto plot_image = result.plot();
        wrt.write(plot_image);
        
        auto end              = std::chrono::system_clock::now();
        auto time_per_frame   = std::chrono::milliseconds(1000 / out_fps);   // ms/frame
        auto time_cost        = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // wait
        if (time_per_frame > time_cost) {
            std::this_thread::sleep_for(time_per_frame - time_cost);
        }
    }
}