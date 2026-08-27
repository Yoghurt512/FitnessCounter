#include <opencv2/opencv.hpp>
#include <iostream>

#include "PoseDetector.h"

int main()
{
    PoseDetector detector;

    if (!detector.initialize(
        "A/models/yolo11n-pose.onnx"))
    {
        return -1;
    }

    // 摄像头：
    cv::VideoCapture cap(0);

    // 如果要测试视频，可改为：
     //cv::VideoCapture cap(
     //    "C:/Users/Y9000P/Desktop/pushvideo/test1.mp4"
     //);

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
        // A 模块
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

        const int key =
            cv::waitKey(1);

        if (key == 27) // ESC
        {
            break;
        }

        if (key == ' ') // Space
        {
            if (detector.confirmCurrentCandidate())
            {
                std::cout
                    << "MAIN PERSON confirmed by SPACE."
                    << std::endl;
            }
            else
            {
                std::cout
                    << "No valid candidate to confirm."
                    << std::endl;
            }
        }

        if (key == 'r' ||
            key == 'R')
        {
            detector.resetMainPerson();
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
