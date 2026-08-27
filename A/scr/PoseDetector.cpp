#include "PoseDetector.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
    // COCO Pose 17 keypoints:
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

        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::cout << "Pose model loaded: " << onnxPath << "\n";
        return true;
    }
    catch (const cv::Exception& e)
    {
        std::cerr
            << "OpenCV failed to load ONNX model:\n"
            << e.what()
            << "\n";

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

bool PoseDetector::confirmCurrentCandidate()
{
    if (mainPersonConfirmed_)
        return true;

    if (!hasSelectionCandidate_)
        return false;

    if (countUsefulKeypoints(selectionCandidate_) < 6)
        return false;

    lockCurrentSelection();
    return true;
}

void PoseDetector::resetMainPerson()
{
    mainPersonConfirmed_ = false;
    hasTrackedPerson_ = false;
    lostFrames_ = 0;

    hasSelectionCandidate_ = false;
    selectionCandidate_ = Candidate{};
    previousTrackedCandidate_ = Candidate{};
    selectionStartTime_ = std::chrono::steady_clock::time_point{};

    std::cout << "MAIN PERSON reset. Start selecting again.\n";
}

bool PoseDetector::isMainPersonConfirmed() const
{
    return mainPersonConfirmed_;
}

cv::Mat PoseDetector::letterbox(
    const cv::Mat& src,
    float& scale,
    int& padX,
    int& padY) const
{
    const float sx =
        static_cast<float>(inputWidth_) /
        static_cast<float>(src.cols);

    const float sy =
        static_cast<float>(inputHeight_) /
        static_cast<float>(src.rows);

    scale = std::min(sx, sy);

    const int newW =
        static_cast<int>(
            std::round(src.cols * scale)
            );

    const int newH =
        static_cast<int>(
            std::round(src.rows * scale)
            );

    cv::Mat resized;
    cv::resize(
        src,
        resized,
        cv::Size(newW, newH)
    );

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
        cv::Scalar(114, 114, 114)
    );

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
    float ox =
        (x - static_cast<float>(padX)) /
        scale;

    float oy =
        (y - static_cast<float>(padY)) /
        scale;

    ox = clampf(
        ox,
        0.0f,
        static_cast<float>(
            originalSize.width - 1
            )
    );

    oy = clampf(
        oy,
        0.0f,
        static_cast<float>(
            originalSize.height - 1
            )
    );

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

    // Ultralytics COCO pose output:
    // [1, 56, 8400] or [1, 8400, 56].
    if (output.dims != 3)
    {
        std::cerr
            << "Unexpected model output dims: "
            << output.dims
            << "\n";

        return candidates;
    }

    const int d1 = output.size[1];
    const int d2 = output.size[2];

    cv::Mat predictions;

    if (d1 == 56)
    {
        cv::Mat temp(
            56,
            d2,
            CV_32F,
            output.ptr<float>()
        );

        cv::transpose(
            temp,
            predictions
        );
    }
    else if (d2 == 56)
    {
        predictions =
            cv::Mat(
                d1,
                56,
                CV_32F,
                output.ptr<float>()
            ).clone();
    }
    else
    {
        std::cerr
            << "Unexpected pose output shape: ["
            << output.size[0] << ", "
            << d1 << ", "
            << d2 << "]\n";

        std::cerr
            << "This code expects an Ultralytics "
            << "COCO pose model with 56 values per prediction.\n";

        return candidates;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<std::vector<KeyPoint>> allKeypoints;

    for (int i = 0;
        i < predictions.rows;
        ++i)
    {
        const float* p =
            predictions.ptr<float>(i);

        const float cx = p[0];
        const float cy = p[1];
        const float w = p[2];
        const float h = p[3];
        const float personScore = p[4];

        if (personScore <
            personConfidenceThreshold_)
        {
            continue;
        }

        const float x1 =
            cx - w * 0.5f;

        const float y1 =
            cy - h * 0.5f;

        const float x2 =
            cx + w * 0.5f;

        const float y2 =
            cy + h * 0.5f;

        cv::Point2f tl =
            mapBack(
                x1,
                y1,
                scale,
                padX,
                padY,
                originalSize
            );

        cv::Point2f br =
            mapBack(
                x2,
                y2,
                scale,
                padX,
                padY,
                originalSize
            );

        int bx =
            static_cast<int>(tl.x);

        int by =
            static_cast<int>(tl.y);

        int bw =
            std::max(
                1,
                static_cast<int>(
                    br.x - tl.x
                    )
            );

        int bh =
            std::max(
                1,
                static_cast<int>(
                    br.y - tl.y
                    )
            );

        cv::Rect box(
            bx,
            by,
            bw,
            bh
        );

        box &=
            cv::Rect(
                0,
                0,
                originalSize.width,
                originalSize.height
            );

        std::vector<KeyPoint>
            keypoints(17);

        for (int k = 0; k < 17; ++k)
        {
            const int base =
                5 + k * 3;

            const float kx =
                p[base + 0];

            const float ky =
                p[base + 1];

            const float kc =
                p[base + 2];

            KeyPoint kp;

            kp.position =
                mapBack(
                    kx,
                    ky,
                    scale,
                    padX,
                    padY,
                    originalSize
                );

            kp.confidence = kc;

            kp.valid =
                kc >=
                keypointConfidenceThreshold_;

            keypoints[k] = kp;
        }

        boxes.push_back(box);
        scores.push_back(personScore);
        allKeypoints.push_back(
            std::move(keypoints)
        );
    }

    std::vector<int> keep;

    cv::dnn::NMSBoxes(
        boxes,
        scores,
        personConfidenceThreshold_,
        nmsThreshold_,
        keep
    );

    for (int idx : keep)
    {
        Candidate c;

        c.box = boxes[idx];
        c.score = scores[idx];
        c.keypoints =
            allKeypoints[idx];

        candidates.push_back(
            std::move(c)
        );
    }

    return candidates;
}

int PoseDetector::countUsefulKeypoints(
    const Candidate& candidate) const
{
    int count = 0;

    for (const auto& item :
        kWantedJoints)
    {
        const int index =
            item.second;

        if (index < 0 ||
            index >=
            static_cast<int>(
                candidate.keypoints.size()
                ))
        {
            continue;
        }

        if (candidate.keypoints[index].confidence >=
            keypointConfidenceThreshold_)
        {
            ++count;
        }
    }

    return count;
}

int PoseDetector::chooseInitialCandidate(
    const std::vector<Candidate>& candidates,
    const cv::Size& frameSize) const
{
    if (candidates.empty())
        return -1;

    const cv::Point2f frameCenter(
        frameSize.width * 0.5f,
        frameSize.height * 0.5f
    );

    const float frameDiagonal =
        std::sqrt(
            static_cast<float>(
                frameSize.width *
                frameSize.width +
                frameSize.height *
                frameSize.height
                )
        );

    const float frameArea =
        static_cast<float>(
            std::max(
                1,
                frameSize.width *
                frameSize.height
            )
            );

    int bestIndex = -1;

    float bestScore =
        -std::numeric_limits<float>::infinity();

    for (int i = 0;
        i < static_cast<int>(
            candidates.size()
            );
        ++i)
    {
        const Candidate& c =
            candidates[i];

        // 选择阶段只考虑关键点足够完整的人。
        if (countUsefulKeypoints(c) < 6)
            continue;

        const cv::Point2f center(
            c.box.x +
            c.box.width * 0.5f,

            c.box.y +
            c.box.height * 0.5f
        );

        const float dx =
            center.x - frameCenter.x;

        const float dy =
            center.y - frameCenter.y;

        const float centerDistance =
            std::sqrt(
                dx * dx +
                dy * dy
            );

        const float centerRatio =
            frameDiagonal > 1.0f
            ? centerDistance /
            frameDiagonal
            : 1.0f;

        const float centerScore =
            clampf(
                1.0f -
                centerRatio * 2.2f,
                0.0f,
                1.0f
            );

        const float areaRatio =
            static_cast<float>(
                c.box.area()
                ) /
            frameArea;

        const float sizeScore =
            clampf(
                std::sqrt(
                    std::max(
                        0.0f,
                        areaRatio
                    )
                ) * 2.0f,
                0.0f,
                1.0f
            );

        // 优先画面中央，同时略偏向更完整、更靠前的人。
        const float score =
            0.70f * centerScore +
            0.20f * sizeScore +
            0.10f * c.score;

        if (score > bestScore)
        {
            bestScore = score;
            bestIndex = i;
        }
    }

    return bestIndex;
}

float PoseDetector::calculateIoU(
    const cv::Rect& a,
    const cv::Rect& b)
{
    const cv::Rect intersection =
        a & b;

    const float intersectionArea =
        static_cast<float>(
            intersection.area()
            );

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

    return intersectionArea /
        unionArea;
}

float PoseDetector::calculateCenterDistanceRatio(
    const cv::Rect& reference,
    const cv::Rect& current)
{
    const cv::Point2f a(
        reference.x +
        reference.width * 0.5f,

        reference.y +
        reference.height * 0.5f
    );

    const cv::Point2f b(
        current.x +
        current.width * 0.5f,

        current.y +
        current.height * 0.5f
    );

    const float dx =
        b.x - a.x;

    const float dy =
        b.y - a.y;

    const float distance =
        std::sqrt(
            dx * dx +
            dy * dy
        );

    const float diagonal =
        std::sqrt(
            static_cast<float>(
                reference.width *
                reference.width +
                reference.height *
                reference.height
                )
        );

    if (diagonal <= 1.0f)
        return 999.0f;

    return distance /
        diagonal;
}

float PoseDetector::calculateSizeSimilarity(
    const cv::Rect& a,
    const cv::Rect& b)
{
    const float areaA =
        static_cast<float>(
            std::max(1, a.area())
            );

    const float areaB =
        static_cast<float>(
            std::max(1, b.area())
            );

    return std::min(
        areaA,
        areaB
    ) /
        std::max(
            areaA,
            areaB
        );
}

float PoseDetector::keypointSimilarity(
    const Candidate& reference,
    const Candidate& current) const
{
    const int count =
        std::min(
            static_cast<int>(
                reference.keypoints.size()
                ),
            static_cast<int>(
                current.keypoints.size()
                )
        );

    if (count <= 0)
        return 0.0f;

    const float referenceDiagonal =
        std::sqrt(
            static_cast<float>(
                reference.box.width *
                reference.box.width +
                reference.box.height *
                reference.box.height
                )
        );

    if (referenceDiagonal <= 1.0f)
        return 0.0f;

    float similaritySum = 0.0f;
    int used = 0;

    for (const auto& item :
        kWantedJoints)
    {
        const int index =
            item.second;

        if (index < 0 ||
            index >= count)
        {
            continue;
        }

        const KeyPoint& a =
            reference.keypoints[index];

        const KeyPoint& b =
            current.keypoints[index];

        if (a.confidence <
            keypointConfidenceThreshold_ ||
            b.confidence <
            keypointConfidenceThreshold_)
        {
            continue;
        }

        const float dx =
            b.position.x -
            a.position.x;

        const float dy =
            b.position.y -
            a.position.y;

        const float distanceRatio =
            std::sqrt(
                dx * dx +
                dy * dy
            ) /
            referenceDiagonal;

        const float similarity =
            clampf(
                1.0f -
                distanceRatio / 0.55f,
                0.0f,
                1.0f
            );

        similaritySum +=
            similarity;

        ++used;
    }

    if (used < 3)
        return 0.0f;

    return similaritySum /
        static_cast<float>(used);
}

bool PoseDetector::passesHardGate(
    const Candidate& reference,
    const Candidate& current,
    bool hardLock) const
{
    if (reference.box.area() <= 0 ||
        current.box.area() <= 0)
    {
        return false;
    }

    const float referenceArea =
        static_cast<float>(
            reference.box.area()
            );

    const float currentArea =
        static_cast<float>(
            current.box.area()
            );

    const float areaRatio =
        currentArea /
        referenceArea;

    // MAIN PERSON 已确认后使用更严格的尺寸连续性。
    const float minAreaRatio =
        hardLock ? 0.50f : 0.42f;

    const float maxAreaRatio =
        hardLock ? 1.80f : 2.20f;

    if (areaRatio < minAreaRatio ||
        areaRatio > maxAreaRatio)
    {
        return false;
    }

    const float centerRatio =
        calculateCenterDistanceRatio(
            reference.box,
            current.box
        );

    const float maxCenterRatio =
        hardLock ? 1.05f : 1.35f;

    if (centerRatio >
        maxCenterRatio)
    {
        return false;
    }

    const float iou =
        calculateIoU(
            reference.box,
            current.box
        );

    // 完全没有框重叠时，中心也必须足够接近。
    if (iou <= 0.001f &&
        centerRatio >
        (hardLock ? 0.62f : 0.82f))
    {
        return false;
    }

    return true;
}

float PoseDetector::candidateMatchScore(
    const Candidate& reference,
    const Candidate& current) const
{
    const float iou =
        calculateIoU(
            reference.box,
            current.box
        );

    const float centerRatio =
        calculateCenterDistanceRatio(
            reference.box,
            current.box
        );

    const float centerSimilarity =
        clampf(
            1.0f -
            centerRatio / 1.15f,
            0.0f,
            1.0f
        );

    const float sizeSimilarity =
        calculateSizeSimilarity(
            reference.box,
            current.box
        );

    const float poseSimilarity =
        keypointSimilarity(
            reference,
            current
        );

    return
        0.42f * iou +
        0.28f * centerSimilarity +
        0.15f * sizeSimilarity +
        0.10f * poseSimilarity +
        0.05f * current.score;
}

int PoseDetector::findBestMatch(
    const Candidate& reference,
    const std::vector<Candidate>& candidates,
    bool hardLock) const
{
    int bestIndex = -1;

    float bestScore =
        -std::numeric_limits<float>::infinity();

    for (int i = 0;
        i < static_cast<int>(
            candidates.size()
            );
        ++i)
    {
        const Candidate& current =
            candidates[i];

        if (countUsefulKeypoints(current) < 6)
            continue;

        if (!passesHardGate(
            reference,
            current,
            hardLock))
        {
            continue;
        }

        const float score =
            candidateMatchScore(
                reference,
                current
            );

        const float minScore =
            hardLock
            ? trackingMatchThreshold_
            : 0.25f;

        if (score < minScore)
            continue;

        if (score > bestScore)
        {
            bestScore = score;
            bestIndex = i;
        }
    }

    return bestIndex;
}

BodyPose PoseDetector::makeBodyPose(
    const Candidate& candidate,
    bool usableByNextModule) const
{
    BodyPose result;

    result.personBox =
        candidate.box;

    int validCount = 0;

    for (const auto& item :
        kWantedJoints)
    {
        const Joint joint =
            item.first;

        const int cocoIndex =
            item.second;

        if (cocoIndex < 0 ||
            cocoIndex >=
            static_cast<int>(
                candidate.keypoints.size()
                ))
        {
            continue;
        }

        KeyPoint kp =
            candidate.keypoints[cocoIndex];

        kp.valid =
            kp.confidence >=
            keypointConfidenceThreshold_;

        result.joints[joint] =
            kp;

        if (kp.valid)
            ++validCount;
    }

    // 选择阶段不把候选人交给 B 模块。
    result.valid =
        usableByNextModule &&
        validCount >= 6;

    return result;
}

void PoseDetector::startSelection(
    const Candidate& candidate)
{
    selectionCandidate_ =
        candidate;

    hasSelectionCandidate_ =
        true;

    selectionStartTime_ =
        std::chrono::steady_clock::now();
}

void PoseDetector::lockCurrentSelection()
{
    if (!hasSelectionCandidate_)
        return;

    mainPersonConfirmed_ =
        true;

    hasTrackedPerson_ =
        true;

    previousTrackedCandidate_ =
        selectionCandidate_;

    lostFrames_ = 0;

    std::cout
        << "MAIN PERSON confirmed.\n";
}

BodyPose PoseDetector::selectAndFilter(
    const std::vector<Candidate>& candidates,
    const cv::Size& frameSize)
{
    BodyPose emptyPose;

    // =========================================================
    // A. MAIN PERSON 尚未确认：3 秒稳定选择 / Space 手动确认
    // =========================================================
    if (!mainPersonConfirmed_)
    {
        if (candidates.empty())
        {
            hasSelectionCandidate_ =
                false;

            selectionCandidate_ =
                Candidate{};

            selectionStartTime_ =
                std::chrono::steady_clock::time_point{};

            return emptyPose;
        }

        int selectedIndex = -1;

        if (!hasSelectionCandidate_)
        {
            selectedIndex =
                chooseInitialCandidate(
                    candidates,
                    frameSize
                );

            if (selectedIndex < 0)
                return emptyPose;

            startSelection(
                candidates[selectedIndex]
            );
        }
        else
        {
            selectedIndex =
                findBestMatch(
                    selectionCandidate_,
                    candidates,
                    false
                );

            // 原候选不再连续：重新选择，并重新从 3 秒开始计时。
            if (selectedIndex < 0)
            {
                selectedIndex =
                    chooseInitialCandidate(
                        candidates,
                        frameSize
                    );

                if (selectedIndex < 0)
                {
                    hasSelectionCandidate_ =
                        false;

                    return emptyPose;
                }

                startSelection(
                    candidates[selectedIndex]
                );
            }
            else
            {
                selectionCandidate_ =
                    candidates[selectedIndex];
            }
        }

        if (!hasSelectionCandidate_)
            return emptyPose;

        const auto now =
            std::chrono::steady_clock::now();

        const double elapsed =
            std::chrono::duration<double>(
                now - selectionStartTime_
            ).count();

        if (elapsed >=
            autoConfirmSeconds_)
        {
            lockCurrentSelection();

            return makeBodyPose(
                previousTrackedCandidate_,
                true
            );
        }

        // 选择阶段返回候选人的关节用于画面显示，
        // 但 valid=false，后面的 B 模块不会提前使用。
        return makeBodyPose(
            selectionCandidate_,
            false
        );
    }

    // =========================================================
    // B. MAIN PERSON 已确认：只跟踪已确认的人，不再选最大路人
    // =========================================================
    if (candidates.empty())
    {
        ++lostFrames_;
        return emptyPose;
    }

    if (!hasTrackedPerson_)
    {
        // 正常情况下不会进入这里。
        // 为避免自动换人，不做任何重新选最大框。
        ++lostFrames_;
        return emptyPose;
    }

    const int selectedIndex =
        findBestMatch(
            previousTrackedCandidate_,
            candidates,
            true
        );

    if (selectedIndex < 0)
    {
        ++lostFrames_;

        // 关键规则：
        // 找不到原 MAIN PERSON 时宁可本帧无姿态，
        // 也绝不自动把旁边的人改成 MAIN PERSON。
        return emptyPose;
    }

    previousTrackedCandidate_ =
        candidates[selectedIndex];

    lostFrames_ = 0;

    return makeBodyPose(
        previousTrackedCandidate_,
        true
    );
}

BodyPose PoseDetector::detect(
    const cv::Mat& frame)
{
    BodyPose emptyPose;

    if (frame.empty() ||
        net_.empty())
    {
        return emptyPose;
    }

    float scale = 1.0f;
    int padX = 0;
    int padY = 0;

    cv::Mat input =
        letterbox(
            frame,
            scale,
            padX,
            padY
        );

    cv::Mat blob =
        cv::dnn::blobFromImage(
            input,
            1.0 / 255.0,
            cv::Size(
                inputWidth_,
                inputHeight_
            ),
            cv::Scalar(),
            true,   // BGR -> RGB
            false,  // no crop
            CV_32F
        );

    net_.setInput(blob);

    std::vector<cv::Mat> outputs;

    const std::vector<std::string>
        outputNames =
        net_.getUnconnectedOutLayersNames();

    net_.forward(
        outputs,
        outputNames
    );

    if (outputs.empty())
        return emptyPose;

    const std::vector<Candidate>
        candidates =
        decode(
            outputs[0],
            frame.size(),
            scale,
            padX,
            padY
        );

    return selectAndFilter(
        candidates,
        frame.size()
    );
}

void PoseDetector::drawPose(
    cv::Mat& frame,
    const BodyPose& pose) const
{
    const std::vector<
        std::pair<Joint, Joint>
    > skeleton =
    {
        {Joint::LeftShoulder,  Joint::LeftElbow},
        {Joint::LeftElbow,     Joint::LeftWrist},

        {Joint::RightShoulder, Joint::RightElbow},
        {Joint::RightElbow,    Joint::RightWrist},

        {Joint::LeftShoulder,  Joint::LeftHip},
        {Joint::RightShoulder, Joint::RightHip},

        {Joint::LeftHip,       Joint::RightHip},

        {Joint::LeftHip,       Joint::LeftKnee},
        {Joint::LeftKnee,      Joint::LeftAnkle},

        {Joint::RightHip,      Joint::RightKnee},
        {Joint::RightKnee,     Joint::RightAnkle}
    };

    // -------------------------
    // 普通骨架
    // -------------------------
    for (const auto& bone :
        skeleton)
    {
        const auto it1 =
            pose.joints.find(
                bone.first
            );

        const auto it2 =
            pose.joints.find(
                bone.second
            );

        if (it1 ==
            pose.joints.end() ||
            it2 ==
            pose.joints.end())
        {
            continue;
        }

        const KeyPoint& kp1 =
            it1->second;

        const KeyPoint& kp2 =
            it2->second;

        if (!kp1.valid ||
            !kp2.valid)
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

    // -------------------------
    // Head -> Neck -> Shoulders
    // -------------------------
    const auto leftShoulderIt =
        pose.joints.find(
            Joint::LeftShoulder
        );

    const auto rightShoulderIt =
        pose.joints.find(
            Joint::RightShoulder
        );

    if (leftShoulderIt !=
        pose.joints.end() &&
        rightShoulderIt !=
        pose.joints.end())
    {
        const KeyPoint& leftShoulder =
            leftShoulderIt->second;

        const KeyPoint& rightShoulder =
            rightShoulderIt->second;

        if (leftShoulder.valid &&
            rightShoulder.valid)
        {
            const cv::Point2f neck =
                (leftShoulder.position +
                    rightShoulder.position)
                * 0.5f;

            cv::line(
                frame,
                neck,
                leftShoulder.position,
                cv::Scalar(0, 255, 255),
                2,
                cv::LINE_AA
            );

            cv::line(
                frame,
                neck,
                rightShoulder.position,
                cv::Scalar(0, 255, 255),
                2,
                cv::LINE_AA
            );

            cv::circle(
                frame,
                neck,
                5,
                cv::Scalar(255, 255, 0),
                -1,
                cv::LINE_AA
            );

            const auto headIt =
                pose.joints.find(
                    Joint::Head
                );

            if (headIt !=
                pose.joints.end())
            {
                const KeyPoint& head =
                    headIt->second;

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

    // -------------------------
    // 关键点 + 名称 + confidence
    // -------------------------
    for (const auto& item :
        pose.joints)
    {
        const Joint joint =
            item.first;

        const KeyPoint& kp =
            item.second;

        if (!kp.valid)
            continue;

        cv::circle(
            frame,
            kp.position,
            5,
            cv::Scalar(0, 255, 0),
            -1,
            cv::LINE_AA
        );

        const std::string text =
            std::string(
                JointName(joint)
            ) +
            " " +
            cv::format(
                "%.2f",
                kp.confidence
            );

        const cv::Point textPosition(
            static_cast<int>(
                kp.position.x + 6
                ),
            static_cast<int>(
                kp.position.y - 6
                )
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

    // =========================================================
    // 状态显示
    // =========================================================
    if (!mainPersonConfirmed_)
    {
        if (hasSelectionCandidate_ &&
            pose.personBox.area() > 0)
        {
            const cv::Scalar orange(
                0,
                165,
                255
            );

            cv::rectangle(
                frame,
                pose.personBox,
                orange,
                3,
                cv::LINE_AA
            );

            const auto now =
                std::chrono::steady_clock::now();

            const double elapsed =
                std::chrono::duration<double>(
                    now -
                    selectionStartTime_
                ).count();

            const double remaining =
                std::max(
                    0.0,
                    autoConfirmSeconds_ -
                    elapsed
                );

            const std::string label =
                std::string(
                    "SELECTING "
                ) +
                cv::format(
                    "%.1fs",
                    remaining
                ) +
                "  [SPACE confirm]";

            const cv::Point labelPos(
                pose.personBox.x,
                std::max(
                    25,
                    pose.personBox.y - 10
                )
            );

            cv::putText(
                frame,
                label,
                labelPos,
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                orange,
                2,
                cv::LINE_AA
            );
        }
        else
        {
            cv::putText(
                frame,
                "Stand near the center - waiting for candidate",
                cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(0, 165, 255),
                2,
                cv::LINE_AA
            );
        }

        cv::putText(
            frame,
            "SPACE: confirm   R: reselect   ESC: exit",
            cv::Point(
                20,
                std::max(
                    30,
                    frame.rows - 20
                )
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA
        );

        return;
    }

    // MAIN PERSON 已确认且当前帧匹配成功
    if (pose.valid &&
        pose.personBox.area() > 0)
    {
        const cv::Scalar green(
            0,
            255,
            0
        );

        cv::rectangle(
            frame,
            pose.personBox,
            green,
            3,
            cv::LINE_AA
        );

        const cv::Point labelPos(
            pose.personBox.x,
            std::max(
                25,
                pose.personBox.y - 10
            )
        );

        cv::putText(
            frame,
            "MAIN PERSON",
            labelPos,
            cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            green,
            2,
            cv::LINE_AA
        );
    }
    else
    {
        // 锁仍然保留，只是当前帧没有找到原来的人。
        cv::putText(
            frame,
            "MAIN PERSON LOST - waiting for the same person",
            cv::Point(20, 35),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(0, 0, 255),
            2,
            cv::LINE_AA
        );
    }

    cv::putText(
        frame,
        "R: reselect MAIN PERSON   ESC: exit",
        cv::Point(
            20,
            std::max(
                30,
                frame.rows - 20
            )
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA
    );
}
