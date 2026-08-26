#include <opencv2/opencv.hpp>
#include <iostream>

#include "PoseDetector.h"

int main()
{
    PoseDetector detector;

    if (!detector.initialize(
        "models/yolo11n-pose.onnx"))
    {
        return -1;
    }

  // cv::VideoCapture cap("C:/Users/Y9000P/Desktop/pushvideo/test1.mp4");
   cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr
            << "Camera open failed."
            << std::endl;

        return -1;
    }

    cv::Mat frame;

    while (true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        // =====================
        // A模块
        // =====================

        BodyPose pose =
            detector.detect(frame);

        detector.drawPose(
            frame,
            pose
        );

        // =====================

        cv::imshow(
            "Fitness Pose",
            frame
        );

        if (cv::waitKey(1) == 27)
            break;
    }

    return 0;
}
