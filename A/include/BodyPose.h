#pragma once

#include <opencv2/core.hpp>

#include <map>
#include <string>


// ==============================
// 我们真正需要的人体关节
// ==============================

enum class Joint
{
    Head,

    LeftShoulder,
    RightShoulder,

    LeftElbow,
    RightElbow,

    LeftWrist,
    RightWrist,

    LeftHip,
    RightHip,

    LeftKnee,
    RightKnee,

    LeftAnkle,
    RightAnkle
};


// ==============================
// 单个人体关键点
// ==============================

struct KeyPoint
{
    // 图像中的像素坐标
    cv::Point2f position{ 0.0f, 0.0f };

    // 模型给出的置信度
    float confidence = 0.0f;

    // 是否通过置信度筛选
    bool valid = false;
};


// ==============================
// A 模块最终输出
// ==============================

struct BodyPose
{
    // 关节 -> 关键点
    std::map<Joint, KeyPoint> joints;

    // 整体姿态是否有效
    bool valid = false;

    // 人体检测框
    cv::Rect personBox;
};


// ==============================
// 关节名称
// 用于画面显示
// ==============================

inline const char* JointName(Joint joint)
{
    switch (joint)
    {
    case Joint::Head:
        return "Head";

    case Joint::LeftShoulder:
        return "LShoulder";

    case Joint::RightShoulder:
        return "RShoulder";

    case Joint::LeftElbow:
        return "LElbow";

    case Joint::RightElbow:
        return "RElbow";

    case Joint::LeftWrist:
        return "LWrist";

    case Joint::RightWrist:
        return "RWrist";

    case Joint::LeftHip:
        return "LHip";

    case Joint::RightHip:
        return "RHip";

    case Joint::LeftKnee:
        return "LKnee";

    case Joint::RightKnee:
        return "RKnee";

    case Joint::LeftAnkle:
        return "LAnkle";

    case Joint::RightAnkle:
        return "RAnkle";

    default:
        return "Unknown";
    }
}
