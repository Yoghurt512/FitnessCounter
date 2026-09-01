#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <chrono>
#include <string>
#include <vector>

#include "BodyPose.h"

class PoseDetector
{
public:
    PoseDetector();

    bool initialize(const std::string& onnxPath);

    // 输入一帧图像。
    // MAIN PERSON 未确认时返回的 BodyPose.valid 为 false，
    // 但 drawPose() 仍会显示橙色候选框。
    BodyPose detect(const cv::Mat& frame);

    void drawPose(
        cv::Mat& frame,
        const BodyPose& pose
    ) const;

    void setPersonConfidenceThreshold(float value);
    void setKeypointConfidenceThreshold(float value);

    // Space：立即确认当前橙色候选人为 MAIN PERSON。
    // 当前没有有效候选人时返回 false。
    bool confirmCurrentCandidate();

    // R：清除当前 MAIN PERSON，重新进入 3 秒选择阶段。
    void resetMainPerson();

    bool isMainPersonConfirmed() const;

private:
    struct Candidate
    {
        cv::Rect box;
        float score = 0.0f;
        std::vector<KeyPoint> keypoints;
    };

    cv::dnn::Net net_;

    int inputWidth_ = 640;
    int inputHeight_ = 640;

    float personConfidenceThreshold_ = 0.40f;
    float nmsThreshold_ = 0.45f;
    float keypointConfidenceThreshold_ = 0.50f;

    // -------------------------
    // MAIN PERSON 选择参数
    // -------------------------
    double autoConfirmSeconds_ = 3.0;

    // -------------------------
    // 已确认后的硬锁定参数
    // -------------------------
    float trackingMatchThreshold_ = 0.34f;

    // 当前 3 秒选择候选
    bool hasSelectionCandidate_ = false;
    Candidate selectionCandidate_;
    std::chrono::steady_clock::time_point selectionStartTime_{};

    // 已确认的 MAIN PERSON 跟踪状态
    bool mainPersonConfirmed_ = false;
    bool hasTrackedPerson_ = false;
    Candidate previousTrackedCandidate_;
    int lostFrames_ = 0;

    cv::Mat letterbox(
        const cv::Mat& src,
        float& scale,
        int& padX,
        int& padY
    ) const;

    std::vector<Candidate> decode(
        const cv::Mat& output,
        const cv::Size& originalSize,
        float scale,
        int padX,
        int padY
    ) const;

    BodyPose selectAndFilter(
        const std::vector<Candidate>& candidates,
        const cv::Size& frameSize
    );

    BodyPose makeBodyPose(
        const Candidate& candidate,
        bool usableByNextModule
    ) const;

    int chooseInitialCandidate(
        const std::vector<Candidate>& candidates,
        const cv::Size& frameSize
    ) const;

    int findBestMatch(
        const Candidate& reference,
        const std::vector<Candidate>& candidates,
        bool hardLock
    ) const;

    bool passesHardGate(
        const Candidate& reference,
        const Candidate& current,
        bool hardLock
    ) const;

    float candidateMatchScore(
        const Candidate& reference,
        const Candidate& current
    ) const;

    float keypointSimilarity(
        const Candidate& reference,
        const Candidate& current
    ) const;

    int countUsefulKeypoints(
        const Candidate& candidate
    ) const;

    void startSelection(
        const Candidate& candidate
    );

    void lockCurrentSelection();

    static float calculateIoU(
        const cv::Rect& a,
        const cv::Rect& b
    );

    static float calculateCenterDistanceRatio(
        const cv::Rect& reference,
        const cv::Rect& current
    );

    static float calculateSizeSimilarity(
        const cv::Rect& a,
        const cv::Rect& b
    );

    static cv::Point2f mapBack(
        float x,
        float y,
        float scale,
        int padX,
        int padY,
        const cv::Size& originalSize
    );
};
