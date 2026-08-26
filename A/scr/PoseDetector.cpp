#include "PoseDetector.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
    // COCO Pose 17 keypoints used by YOLO pose models:
    // 0 nose
    // 1 left_eye
    // 2 right_eye
    // 3 left_ear
    // 4 right_ear
    // 5 left_shoulder
    // 6 right_shoulder
    // 7 left_elbow
    // 8 right_elbow
    // 9 left_wrist
    // 10 right_wrist
    // 11 left_hip
    // 12 right_hip
    // 13 left_knee
    // 14 right_knee
    // 15 left_ankle
    // 16 right_ankle

    const std::vector<std::pair<Joint, int>> kWantedJoints =
    {
        {Joint::Head,          0},
        {Joint::LeftShoulder,  5},
        {Joint::RightShoulder, 6},
        {Joint::LeftElbow,     7},
        {Joint::RightElbow,    8},
        {Joint::LeftWrist,     9},
        {Joint::RightWrist,   10},
        {Joint::LeftHip,      11},
        {Joint::RightHip,     12},
        {Joint::LeftKnee,     13},
        {Joint::RightKnee,    14},
        {Joint::LeftAnkle,    15},
        {Joint::RightAnkle,   16}
    };

    float clampf(float v, float lo, float hi)
    {
        return std::max(lo, std::min(v, hi));
    }
}

PoseDetector::PoseDetector() = default;

bool PoseDetector::initialize(const std::string& onnxPath)
{
    try
    {
        net_ = cv::dnn::readNetFromONNX(onnxPath);
        if (net_.empty())
        {
            std::cerr << "ERROR: ONNX model is empty.\n";
            return false;
        }

        // CPU is the safest default for a first working version.
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::cout << "Pose model loaded: " << onnxPath << "\n";
        return true;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "OpenCV failed to load ONNX model:\n"
            << e.what() << "\n";
        return false;
    }
}

void PoseDetector::setPersonConfidenceThreshold(float value)
{
    personConfidenceThreshold_ = value;
}

void PoseDetector::setKeypointConfidenceThreshold(float value)
{
    keypointConfidenceThreshold_ = value;
}


void PoseDetector::resetTracking()
{
    hasTrackedPerson_ = false;
    previousPersonBox_ = cv::Rect();
    lostFrames_ = 0;
}

float PoseDetector::calculateIoU(
    const cv::Rect& a,
    const cv::Rect& b)
{
    const cv::Rect intersection = a & b;

    const float intersectionArea =
        static_cast<float>(intersection.area());

    if (intersectionArea <= 0.0f)
        return 0.0f;

    const float unionArea =
        static_cast<float>(
            a.area() +
            b.area() -
            intersection.area()
            );

    if (unionArea <= 0.0f)
        return 0.0f;

    return intersectionArea / unionArea;
}

float PoseDetector::calculateCenterSimilarity(
    const cv::Rect& a,
    const cv::Rect& b,
    const cv::Size& frameSize)
{
    const cv::Point2f centerA(
        a.x + a.width * 0.5f,
        a.y + a.height * 0.5f
    );

    const cv::Point2f centerB(
        b.x + b.width * 0.5f,
        b.y + b.height * 0.5f
    );

    const float dx = centerA.x - centerB.x;
    const float dy = centerA.y - centerB.y;

    const float distance =
        std::sqrt(dx * dx + dy * dy);

    const float fw = static_cast<float>(frameSize.width);
    const float fh = static_cast<float>(frameSize.height);

    const float diagonal =
        std::sqrt(fw * fw + fh * fh);

    if (diagonal <= 0.0f)
        return 0.0f;

    const float normalizedDistance =
        std::min(distance / diagonal, 1.0f);

    return 1.0f - normalizedDistance;
}

float PoseDetector::calculateSizeSimilarity(
    const cv::Rect& a,
    const cv::Rect& b)
{
    const float areaA =
        static_cast<float>(a.area());

    const float areaB =
        static_cast<float>(b.area());

    if (areaA <= 0.0f || areaB <= 0.0f)
        return 0.0f;

    return std::min(areaA, areaB) /
        std::max(areaA, areaB);
}

cv::Mat PoseDetector::letterbox(
    const cv::Mat& src,
    float& scale,
    int& padX,
    int& padY) const
{
    const float sx = static_cast<float>(inputWidth_) / src.cols;
    const float sy = static_cast<float>(inputHeight_) / src.rows;
    scale = std::min(sx, sy);

    const int newW = static_cast<int>(std::round(src.cols * scale));
    const int newH = static_cast<int>(std::round(src.rows * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));

    const int dw = inputWidth_ - newW;
    const int dh = inputHeight_ - newH;

    const int left = dw / 2;
    const int right = dw - left;
    const int top = dh / 2;
    const int bottom = dh - top;

    padX = left;
    padY = top;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

    return padded;
}

cv::Point2f PoseDetector::mapBack(
    float x,
    float y,
    float scale,
    int padX,
    int padY,
    const cv::Size& originalSize)
{
    float ox = (x - padX) / scale;
    float oy = (y - padY) / scale;

    ox = clampf(ox, 0.0f, static_cast<float>(originalSize.width - 1));
    oy = clampf(oy, 0.0f, static_cast<float>(originalSize.height - 1));

    return cv::Point2f(ox, oy);
}

std::vector<PoseDetector::Candidate> PoseDetector::decode(
    const cv::Mat& rawOutput,
    const cv::Size& originalSize,
    float scale,
    int padX,
    int padY) const
{
    std::vector<Candidate> candidates;

    if (rawOutput.empty())
        return candidates;

    cv::Mat output = rawOutput;

    // Typical exported Ultralytics pose output is:
    // [1, 56, 8400]  -> 4 box + 1 person score + 17*3 keypoint values
    //
    // Some exports can appear as [1, 8400, 56].
    // Normalize both layouts to an Nx56 matrix.

    if (output.dims != 3)
    {
        std::cerr << "Unexpected model output dims: " << output.dims << "\n";
        return candidates;
    }

    const int d1 = output.size[1];
    const int d2 = output.size[2];

    cv::Mat predictions;

    if (d1 == 56)
    {
        cv::Mat temp(56, d2, CV_32F, output.ptr<float>());
        cv::transpose(temp, predictions); // d2 x 56
    }
    else if (d2 == 56)
    {
        predictions = cv::Mat(d1, 56, CV_32F, output.ptr<float>()).clone();
    }
    else
    {
        std::cerr << "Unexpected pose output shape: ["
            << output.size[0] << ", "
            << d1 << ", "
            << d2 << "]\n";
        std::cerr << "This code expects an Ultralytics COCO pose model with 56 values per prediction.\n";
        return candidates;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::vector<KeyPoint>> allKeypoints;

    for (int i = 0; i < predictions.rows; ++i)
    {
        const float* p = predictions.ptr<float>(i);

        const float cx = p[0];
        const float cy = p[1];
        const float w = p[2];
        const float h = p[3];
        const float personScore = p[4];

        if (personScore < personConfidenceThreshold_)
            continue;

        const float x1 = cx - w * 0.5f;
        const float y1 = cy - h * 0.5f;
        const float x2 = cx + w * 0.5f;
        const float y2 = cy + h * 0.5f;

        cv::Point2f tl = mapBack(x1, y1, scale, padX, padY, originalSize);
        cv::Point2f br = mapBack(x2, y2, scale, padX, padY, originalSize);

        int bx = static_cast<int>(tl.x);
        int by = static_cast<int>(tl.y);
        int bw = std::max(1, static_cast<int>(br.x - tl.x));
        int bh = std::max(1, static_cast<int>(br.y - tl.y));

        cv::Rect box(bx, by, bw, bh);
        box &= cv::Rect(0, 0, originalSize.width, originalSize.height);

        std::vector<KeyPoint> keypoints(17);

        for (int k = 0; k < 17; ++k)
        {
            const int base = 5 + k * 3;

            const float kx = p[base + 0];
            const float ky = p[base + 1];
            const float kc = p[base + 2];

            KeyPoint kp;
            kp.position = mapBack(kx, ky, scale, padX, padY, originalSize);
            kp.confidence = kc;
            kp.valid = kc >= keypointConfidenceThreshold_;

            keypoints[k] = kp;
        }

        boxes.push_back(box);
        scores.push_back(personScore);
        allKeypoints.push_back(std::move(keypoints));
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(
        boxes,
        scores,
        personConfidenceThreshold_,
        nmsThreshold_,
        keep);

    for (int idx : keep)
    {
        Candidate c;
        c.box = boxes[idx];
        c.score = scores[idx];
        c.keypoints = allKeypoints[idx];
        candidates.push_back(std::move(c));
    }

    return candidates;
}

BodyPose PoseDetector::selectAndFilter(
    const std::vector<Candidate>& candidates,
    const cv::Size& frameSize)
{
    BodyPose result;

    // ==========================================================
    // 1. 当前没有检测到任何人
    // ==========================================================
    if (candidates.empty())
    {
        if (hasTrackedPerson_)
        {
            ++lostFrames_;

            // 可能只是短暂遮挡/漏检，先不要立即切换到别人。
            if (lostFrames_ <= maxLostFrames_)
                return result;

            // 丢失时间过长，解除旧目标锁定。
            resetTracking();
        }

        return result;
    }


    // ==========================================================
    // 2. 选择当前帧主运动者候选
    // ==========================================================
    int selectedIndex = -1;

    // 第一次检测：选择画面中最大的人。
    if (!hasTrackedPerson_)
    {
        int largestArea = -1;

        for (int i = 0;
            i < static_cast<int>(candidates.size());
            ++i)
        {
            const int area = candidates[i].box.area();

            if (area > largestArea)
            {
                largestArea = area;
                selectedIndex = i;
            }
        }
    }
    else
    {
        // 后续帧：优先寻找“最像上一帧主运动者”的人。
        float bestTrackingScore = -1.0f;

        for (int i = 0;
            i < static_cast<int>(candidates.size());
            ++i)
        {
            const Candidate& candidate = candidates[i];

            const float iou =
                calculateIoU(
                    previousPersonBox_,
                    candidate.box
                );

            const float centerSimilarity =
                calculateCenterSimilarity(
                    previousPersonBox_,
                    candidate.box,
                    frameSize
                );

            const float sizeSimilarity =
                calculateSizeSimilarity(
                    previousPersonBox_,
                    candidate.box
                );

            const float trackingScore =
                0.55f * iou +
                0.25f * centerSimilarity +
                0.15f * sizeSimilarity +
                0.05f * candidate.score;

            if (trackingScore > bestTrackingScore)
            {
                bestTrackingScore = trackingScore;
                selectedIndex = i;
            }
        }

        // 最相似的人仍然不够像：先视为主运动者暂时丢失。
        if (bestTrackingScore < trackingMatchThreshold_)
        {
            ++lostFrames_;

            if (lostFrames_ <= maxLostFrames_)
                return result;

            // 连续丢失太久，重新选择当前帧中最大的人。
            resetTracking();

            int largestArea = -1;
            selectedIndex = -1;

            for (int i = 0;
                i < static_cast<int>(candidates.size());
                ++i)
            {
                const int area = candidates[i].box.area();

                if (area > largestArea)
                {
                    largestArea = area;
                    selectedIndex = i;
                }
            }
        }
    }


    // ==========================================================
    // 3. 没有找到有效候选
    // ==========================================================
    if (selectedIndex < 0 ||
        selectedIndex >= static_cast<int>(candidates.size()))
    {
        return result;
    }


    // ==========================================================
    // 4. 更新主运动者跟踪状态
    // ==========================================================
    const Candidate& selected =
        candidates[selectedIndex];

    previousPersonBox_ = selected.box;
    hasTrackedPerson_ = true;
    lostFrames_ = 0;

    result.personBox = selected.box;


    // ==========================================================
    // 5. 保留当前项目需要的关键点
    // ==========================================================
    int validCount = 0;

    for (const auto& item : kWantedJoints)
    {
        const Joint joint = item.first;
        const int cocoIndex = item.second;

        if (cocoIndex < 0 ||
            cocoIndex >= static_cast<int>(selected.keypoints.size()))
        {
            continue;
        }

        KeyPoint kp = selected.keypoints[cocoIndex];

        // 最终再按当前阈值统一过滤一次。
        kp.valid =
            kp.confidence >= keypointConfidenceThreshold_;

        result.joints[joint] = kp;

        if (kp.valid)
            ++validCount;
    }

    result.valid = validCount >= 6;

    return result;
}

BodyPose PoseDetector::detect(const cv::Mat& frame)
{
    BodyPose emptyPose;

    if (frame.empty() || net_.empty())
        return emptyPose;

    float scale = 1.0f;
    int padX = 0;
    int padY = 0;

    cv::Mat input = letterbox(frame, scale, padX, padY);

    cv::Mat blob = cv::dnn::blobFromImage(
        input,
        1.0 / 255.0,
        cv::Size(inputWidth_, inputHeight_),
        cv::Scalar(),
        true,   // BGR -> RGB
        false,  // no crop
        CV_32F);

    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    std::vector<std::string> outputNames = net_.getUnconnectedOutLayersNames();
    net_.forward(outputs, outputNames);

    if (outputs.empty())
        return emptyPose;

    std::vector<Candidate> candidates =
        decode(outputs[0], frame.size(), scale, padX, padY);

    return selectAndFilter(candidates, frame.size());
}

void PoseDetector::drawPose(
    cv::Mat& frame,
    const BodyPose& pose) const
{
    // ==========================================================
    // 1. 定义身体骨架
    //
    // 注意：
    // Head 不直接连接 LeftShoulder / RightShoulder
    //
    // Head -> Neck -> Shoulders
    // 在后面单独绘制
    // ==========================================================

    const std::vector<std::pair<Joint, Joint>> skeleton =
    {
        // 左手臂
        { Joint::LeftShoulder, Joint::LeftElbow },
        { Joint::LeftElbow, Joint::LeftWrist },

        // 右手臂
        { Joint::RightShoulder, Joint::RightElbow },
        { Joint::RightElbow, Joint::RightWrist },

        // 左侧躯干
        { Joint::LeftShoulder, Joint::LeftHip },

        // 右侧躯干
        { Joint::RightShoulder, Joint::RightHip },

        // 髋部连接
        { Joint::LeftHip, Joint::RightHip },

        // 左腿
        { Joint::LeftHip, Joint::LeftKnee },
        { Joint::LeftKnee, Joint::LeftAnkle },

        // 右腿
        { Joint::RightHip, Joint::RightKnee },
        { Joint::RightKnee, Joint::RightAnkle }
    };


    // ==========================================================
    // 2. 绘制普通骨架
    // ==========================================================

    for (const auto& bone : skeleton)
    {
        auto it1 = pose.joints.find(bone.first);
        auto it2 = pose.joints.find(bone.second);

        // 关节不存在
        if (it1 == pose.joints.end() ||
            it2 == pose.joints.end())
        {
            continue;
        }

        const KeyPoint& kp1 = it1->second;
        const KeyPoint& kp2 = it2->second;

        // 两个点必须同时有效
        if (!kp1.valid || !kp2.valid)
        {
            continue;
        }

        cv::line(
            frame,
            kp1.position,
            kp2.position,
            cv::Scalar(0, 255, 255),
            2,
            cv::LINE_AA
        );
    }


    // ==========================================================
    // 3. 计算 Neck
    //
    // Neck 不属于模型关键点
    //
    // Neck = 左肩和右肩的中点
    // ==========================================================

    auto leftShoulderIt =
        pose.joints.find(Joint::LeftShoulder);

    auto rightShoulderIt =
        pose.joints.find(Joint::RightShoulder);


    if (leftShoulderIt != pose.joints.end() &&
        rightShoulderIt != pose.joints.end())
    {
        const KeyPoint& leftShoulder =
            leftShoulderIt->second;

        const KeyPoint& rightShoulder =
            rightShoulderIt->second;


        // 左右肩都有效才能计算 Neck
        if (leftShoulder.valid &&
            rightShoulder.valid)
        {
            // ==========================================
            // Neck = 左右肩中点
            // ==========================================

            cv::Point2f neck =
                (leftShoulder.position +
                    rightShoulder.position) * 0.5f;


            // ==========================================
            // Neck -> LeftShoulder
            // ==========================================

            cv::line(
                frame,
                neck,
                leftShoulder.position,
                cv::Scalar(0, 255, 255),
                2,
                cv::LINE_AA
            );


            // ==========================================
            // Neck -> RightShoulder
            // ==========================================

            cv::line(
                frame,
                neck,
                rightShoulder.position,
                cv::Scalar(0, 255, 255),
                2,
                cv::LINE_AA
            );


            // ==========================================
            // 画 Neck 辅助点
            // ==========================================

            cv::circle(
                frame,
                neck,
                5,
                cv::Scalar(255, 255, 0),
                -1,
                cv::LINE_AA
            );


            // ==========================================
            // 查找 Head
            // ==========================================

            auto headIt =
                pose.joints.find(Joint::Head);

            if (headIt != pose.joints.end())
            {
                const KeyPoint& head =
                    headIt->second;


                // Head 有效才连接 Head -> Neck
                if (head.valid)
                {
                    cv::line(
                        frame,
                        head.position,
                        neck,
                        cv::Scalar(0, 255, 255),
                        2,
                        cv::LINE_AA
                    );
                }
            }
        }
    }


    // ==========================================================
    // 4. 绘制所有真正的关键点
    // ==========================================================

    for (const auto& item : pose.joints)
    {
        Joint joint = item.first;

        const KeyPoint& kp =
            item.second;

        if (!kp.valid)
        {
            continue;
        }


        // ==========================================
        // 绘制关键点
        // ==========================================

        cv::circle(
            frame,
            kp.position,
            5,
            cv::Scalar(0, 255, 0),
            -1,
            cv::LINE_AA
        );


        // ==========================================
        // 名称 + confidence
        // ==========================================

        std::string text =
            std::string(JointName(joint)) +
            " " +
            cv::format("%.2f", kp.confidence);


        cv::Point textPosition(
            static_cast<int>(kp.position.x + 6),
            static_cast<int>(kp.position.y - 6)
        );


        cv::putText(
            frame,
            text,
            textPosition,
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA
        );
    }


    // ==========================================================
    // 5. 绘制人体检测框
    // ==========================================================

    if (pose.valid)
    {
        cv::rectangle(
            frame,
            pose.personBox,
            cv::Scalar(255, 0, 0),
            2,
            cv::LINE_AA
        );

        cv::putText(
            frame,
            "MAIN PERSON",
            cv::Point(
                pose.personBox.x,
                std::max(20, pose.personBox.y - 10)
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 0),
            2,
            cv::LINE_AA
        );
    }
}
