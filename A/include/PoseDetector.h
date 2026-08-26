#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

#include "BodyPose.h"

class PoseDetector
{
public:

    PoseDetector();

    // 加载 ONNX 姿态模型
    bool initialize(const std::string& onnxPath);

    // 输入一帧图像，返回筛选后的 BodyPose
    BodyPose detect(const cv::Mat& frame);

    // 在画面上绘制关键点和骨架
    void drawPose(
        cv::Mat& frame,
        const BodyPose& pose
    ) const;

    // 人体检测置信度阈值
    void setPersonConfidenceThreshold(float value);

    // 关键点置信度阈值
    void setKeypointConfidenceThreshold(float value);

    // 手动清除当前主运动者跟踪状态
    void resetTracking();


private:

    // -------------------------
    // 单个人体候选结果
    // -------------------------
    struct Candidate
    {
        cv::Rect box;

        float score = 0.0f;

        // YOLO Pose 的 COCO 17 个关键点
        std::vector<KeyPoint> keypoints;
    };


    // -------------------------
    // 神经网络
    // -------------------------
    cv::dnn::Net net_;


    // -------------------------
    // 模型输入尺寸
    // -------------------------
    int inputWidth_ = 640;
    int inputHeight_ = 640;


    // -------------------------
    // 检测阈值
    // -------------------------
    float personConfidenceThreshold_ = 0.40f;

    float nmsThreshold_ = 0.45f;

    float keypointConfidenceThreshold_ = 0.50f;


    // -------------------------
    // 主运动者跨帧跟踪状态
    // -------------------------
    bool hasTrackedPerson_ = false;

    cv::Rect previousPersonBox_;

    int lostFrames_ = 0;

    int maxLostFrames_ = 10;

    float trackingMatchThreshold_ = 0.30f;


    // -------------------------
    // 图像等比例缩放 + 填充
    // -------------------------
    cv::Mat letterbox(
        const cv::Mat& src,
        float& scale,
        int& padX,
        int& padY
    ) const;


    // -------------------------
    // 解析 YOLO Pose 输出
    // -------------------------
    std::vector<Candidate> decode(
        const cv::Mat& output,
        const cv::Size& originalSize,
        float scale,
        int padX,
        int padY
    ) const;


    // -------------------------
    // 稳定选择主运动者 + 筛选需要的关节
    //
    // 注意：这里不能再 const，
    // 因为需要更新上一帧主运动者状态
    // -------------------------
    BodyPose selectAndFilter(
        const std::vector<Candidate>& candidates,
        const cv::Size& frameSize
    );


    // 两个人体框的 IoU
    static float calculateIoU(
        const cv::Rect& a,
        const cv::Rect& b
    );

    // 两个人体框中心位置相似度
    static float calculateCenterSimilarity(
        const cv::Rect& a,
        const cv::Rect& b,
        const cv::Size& frameSize
    );

    // 两个人体框尺寸相似度
    static float calculateSizeSimilarity(
        const cv::Rect& a,
        const cv::Rect& b
    );


    // -------------------------
    // 将 640x640 模型坐标映射回原图坐标
    // -------------------------
    static cv::Point2f mapBack(
        float x,
        float y,
        float scale,
        int padX,
        int padY,
        const cv::Size& originalSize
    );
};
