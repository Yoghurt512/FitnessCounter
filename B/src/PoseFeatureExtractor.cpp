#include "PoseFeatureExtractor.h"

#include <algorithm>
#include <cmath>

namespace
{
    constexpr float kRadiansToDegrees = 57.29577951308232f;

    struct SideJoints
    {
        const KeyPoint* shoulder = nullptr;
        const KeyPoint* elbow = nullptr;
        const KeyPoint* wrist = nullptr;
        const KeyPoint* hip = nullptr;
        const KeyPoint* knee = nullptr;
        const KeyPoint* ankle = nullptr;

        int validCount = 0;
        float confidenceSum = 0.0f;
    };

    bool isFinite(float value)
    {
        return std::isfinite(value);
    }

    float clampf(float value, float minimum, float maximum)
    {
        return std::max(minimum, std::min(value, maximum));
    }

    const KeyPoint* findValidJoint(
        const BodyPose& pose,
        Joint joint,
        float minConfidence)
    {
        const auto iterator = pose.joints.find(joint);

        if (iterator == pose.joints.end())
            return nullptr;

        const KeyPoint& keypoint = iterator->second;
        if (!keypoint.valid || keypoint.confidence < minConfidence)
            return nullptr;

        return &keypoint;
    }

    void addJoint(const KeyPoint* keypoint, SideJoints& joints)
    {
        if (keypoint == nullptr)
            return;

        ++joints.validCount;
        joints.confidenceSum += keypoint->confidence;
    }

    SideJoints collectSideJoints(
        const BodyPose& pose,
        BodySide side,
        float minConfidence)
    {
        SideJoints joints;
        const bool left = side == BodySide::Left;

        joints.shoulder = findValidJoint(
            pose,
            left ? Joint::LeftShoulder : Joint::RightShoulder,
            minConfidence);
        joints.elbow = findValidJoint(
            pose,
            left ? Joint::LeftElbow : Joint::RightElbow,
            minConfidence);
        joints.wrist = findValidJoint(
            pose,
            left ? Joint::LeftWrist : Joint::RightWrist,
            minConfidence);
        joints.hip = findValidJoint(
            pose,
            left ? Joint::LeftHip : Joint::RightHip,
            minConfidence);
        joints.knee = findValidJoint(
            pose,
            left ? Joint::LeftKnee : Joint::RightKnee,
            minConfidence);
        joints.ankle = findValidJoint(
            pose,
            left ? Joint::LeftAnkle : Joint::RightAnkle,
            minConfidence);

        addJoint(joints.shoulder, joints);
        addJoint(joints.elbow, joints);
        addJoint(joints.wrist, joints);
        addJoint(joints.hip, joints);
        addJoint(joints.knee, joints);
        addJoint(joints.ankle, joints);

        return joints;
    }

    bool hasUpperChain(const SideJoints& joints)
    {
        return joints.shoulder != nullptr &&
            joints.elbow != nullptr &&
            joints.wrist != nullptr;
    }

    bool hasLowerChain(const SideJoints& joints)
    {
        return joints.hip != nullptr &&
            joints.knee != nullptr &&
            joints.ankle != nullptr;
    }

    float sideScore(const SideJoints& joints)
    {
        if (!hasUpperChain(joints) && !hasLowerChain(joints))
            return -1.0f;

        const float meanConfidence = joints.validCount == 0
            ? 0.0f
            : joints.confidenceSum / joints.validCount;

        return meanConfidence + 0.05f * joints.validCount;
    }

    float calculateAngle(
        const KeyPoint* first,
        const KeyPoint* vertex,
        const KeyPoint* third)
    {
        if (first == nullptr || vertex == nullptr || third == nullptr)
            return std::numeric_limits<float>::quiet_NaN();

        const cv::Point2f firstVector = first->position - vertex->position;
        const cv::Point2f thirdVector = third->position - vertex->position;

        const float firstLength = std::sqrt(firstVector.dot(firstVector));
        const float thirdLength = std::sqrt(thirdVector.dot(thirdVector));

        if (firstLength <= 1e-5f || thirdLength <= 1e-5f)
            return std::numeric_limits<float>::quiet_NaN();

        const float cosine = clampf(
            firstVector.dot(thirdVector) / (firstLength * thirdLength),
            -1.0f,
            1.0f);

        return std::acos(cosine) * kRadiansToDegrees;
    }

    float calculateTorsoTilt(
        const KeyPoint* shoulder,
        const KeyPoint* hip)
    {
        if (shoulder == nullptr || hip == nullptr)
            return std::numeric_limits<float>::quiet_NaN();

        const cv::Point2f torso = shoulder->position - hip->position;
        if (std::abs(torso.x) <= 1e-5f && std::abs(torso.y) <= 1e-5f)
            return std::numeric_limits<float>::quiet_NaN();

        return std::atan2(std::abs(torso.x), std::abs(torso.y)) *
            kRadiansToDegrees;
    }

    float calculateKneeInwardScore(
        const SideJoints& left,
        const SideJoints& right)
    {
        if (left.hip == nullptr || left.knee == nullptr ||
            left.ankle == nullptr || right.hip == nullptr ||
            right.knee == nullptr || right.ankle == nullptr)
        {
            return std::numeric_limits<float>::quiet_NaN();
        }

        const float leftHipX = left.hip->position.x;
        const float rightHipX = right.hip->position.x;
        const float hipWidth = std::abs(rightHipX - leftHipX);
        if (hipWidth < 1e-5f)
            return std::numeric_limits<float>::quiet_NaN();

        const float leftExpected =
            (leftHipX + left.ankle->position.x) * 0.5f;
        const float rightExpected =
            (rightHipX + right.ankle->position.x) * 0.5f;

        // The signs work for either a mirrored or a non-mirrored camera.
        const float leftTowardCenter =
            (leftHipX < rightHipX ? 1.0f : -1.0f) *
            (left.knee->position.x - leftExpected);
        const float rightTowardCenter =
            (rightHipX < leftHipX ? 1.0f : -1.0f) *
            (right.knee->position.x - rightExpected);

        return std::max(0.0f,
            (std::max(0.0f, leftTowardCenter) +
             std::max(0.0f, rightTowardCenter)) /
            (2.0f * hipWidth));
    }

    float averageY(const KeyPoint* first, const KeyPoint* second)
    {
        if (first == nullptr && second == nullptr)
            return std::numeric_limits<float>::quiet_NaN();
        if (first == nullptr) return second->position.y;
        if (second == nullptr) return first->position.y;
        return (first->position.y + second->position.y) * 0.5f;
    }

    float averageX(const KeyPoint* first, const KeyPoint* second)
    {
        if (first == nullptr && second == nullptr)
            return std::numeric_limits<float>::quiet_NaN();
        if (first == nullptr) return second->position.x;
        if (second == nullptr) return first->position.x;
        return (first->position.x + second->position.x) * 0.5f;
    }

    float smoothValue(float previous, float current, float factor)
    {
        if (!isFinite(current))
            return current;

        if (!isFinite(previous))
            return current;

        return previous + factor * (current - previous);
    }

    void smoothFeatures(
        PoseFeatures& current,
        const PoseFeatures& previous,
        float factor)
    {
        current.elbowAngleDeg = smoothValue(
            previous.elbowAngleDeg, current.elbowAngleDeg, factor);
        current.shoulderAngleDeg = smoothValue(
            previous.shoulderAngleDeg, current.shoulderAngleDeg, factor);
        current.hipAngleDeg = smoothValue(
            previous.hipAngleDeg, current.hipAngleDeg, factor);
        current.kneeAngleDeg = smoothValue(
            previous.kneeAngleDeg, current.kneeAngleDeg, factor);
        current.leftHipAngleDeg = smoothValue(
            previous.leftHipAngleDeg, current.leftHipAngleDeg, factor);
        current.rightHipAngleDeg = smoothValue(
            previous.rightHipAngleDeg, current.rightHipAngleDeg, factor);
        current.leftKneeAngleDeg = smoothValue(
            previous.leftKneeAngleDeg, current.leftKneeAngleDeg, factor);
        current.rightKneeAngleDeg = smoothValue(
            previous.rightKneeAngleDeg, current.rightKneeAngleDeg, factor);
        current.torsoTiltDeg = smoothValue(
            previous.torsoTiltDeg, current.torsoTiltDeg, factor);
        current.kneeInwardScore = smoothValue(
            previous.kneeInwardScore, current.kneeInwardScore, factor);
        current.wristShoulderHeight = smoothValue(
            previous.wristShoulderHeight, current.wristShoulderHeight, factor);
        current.armSpreadScore = smoothValue(
            previous.armSpreadScore, current.armSpreadScore, factor);
        current.legSpreadScore = smoothValue(
            previous.legSpreadScore, current.legSpreadScore, factor);
    }
}

PoseFeatureExtractor::PoseFeatureExtractor(
    const PoseFeatureConfig& config)
{
    setConfig(config);
}

PoseFeatures PoseFeatureExtractor::extract(const BodyPose& pose)
{
    PoseFeatures features;

    for (const auto& entry : pose.joints)
    {
        const KeyPoint& keypoint = entry.second;
        if (!keypoint.valid ||
            keypoint.confidence < config_.minKeypointConfidence)
        {
            continue;
        }

        ++features.validKeypointCount;
        features.meanKeypointConfidence += keypoint.confidence;
    }

    if (features.validKeypointCount > 0)
    {
        features.meanKeypointConfidence /= features.validKeypointCount;
    }

    const SideJoints left = collectSideJoints(
        pose, BodySide::Left, config_.minKeypointConfidence);
    const SideJoints right = collectSideJoints(
        pose, BodySide::Right, config_.minKeypointConfidence);

    const float leftScore = sideScore(left);
    const float rightScore = sideScore(right);

    const SideJoints* activeJoints = nullptr;
    if (leftScore >= rightScore && leftScore >= 0.0f)
    {
        features.activeSide = BodySide::Left;
        activeJoints = &left;
    }
    else if (rightScore >= 0.0f)
    {
        features.activeSide = BodySide::Right;
        activeJoints = &right;
    }

    // Calculate both sides independently. The counter can fuse them for a
    // front-facing squat, while still falling back to one side when the other
    // side is occluded in a side view.
    features.leftHipAngleDeg = calculateAngle(
        left.shoulder, left.hip, left.knee);
    features.rightHipAngleDeg = calculateAngle(
        right.shoulder, right.hip, right.knee);
    features.leftKneeAngleDeg = calculateAngle(
        left.hip, left.knee, left.ankle);
    features.rightKneeAngleDeg = calculateAngle(
        right.hip, right.knee, right.ankle);
    features.kneeInwardScore = calculateKneeInwardScore(left, right);

    const float shoulderY = averageY(left.shoulder, right.shoulder);
    const float wristY = averageY(left.wrist, right.wrist);
    const float hipY = averageY(left.hip, right.hip);
    const float ankleY = averageY(left.ankle, right.ankle);
    const float shoulderWidth = std::abs(
        (left.shoulder != nullptr && right.shoulder != nullptr)
            ? right.shoulder->position.x - left.shoulder->position.x : 0.0f);
    const float hipWidth = std::abs(
        (left.hip != nullptr && right.hip != nullptr)
            ? right.hip->position.x - left.hip->position.x : 0.0f);
    if (isFinite(shoulderY) && isFinite(wristY) && isFinite(hipY))
    {
        const float torsoScale = std::max(20.0f, std::abs(hipY - shoulderY));
        features.wristShoulderHeight = (shoulderY - wristY) / torsoScale;
    }
    if (isFinite(shoulderWidth) && shoulderWidth > 1.0f &&
        left.wrist != nullptr && right.wrist != nullptr)
    {
        features.armSpreadScore = std::abs(
            right.wrist->position.x - left.wrist->position.x) / shoulderWidth;
    }
    if (isFinite(hipWidth) && hipWidth > 1.0f &&
        left.ankle != nullptr && right.ankle != nullptr)
    {
        features.legSpreadScore = std::abs(
            right.ankle->position.x - left.ankle->position.x) / hipWidth;
    }

    if (activeJoints != nullptr)
    {
        features.sideConfidence = activeJoints->validCount == 0
            ? 0.0f
            : activeJoints->confidenceSum / activeJoints->validCount;

        features.elbowAngleDeg = calculateAngle(
            activeJoints->shoulder,
            activeJoints->elbow,
            activeJoints->wrist);
        features.shoulderAngleDeg = calculateAngle(
            activeJoints->elbow,
            activeJoints->shoulder,
            activeJoints->hip);
        features.hipAngleDeg = calculateAngle(
            activeJoints->shoulder,
            activeJoints->hip,
            activeJoints->knee);
        features.kneeAngleDeg = calculateAngle(
            activeJoints->hip,
            activeJoints->knee,
            activeJoints->ankle);
        features.torsoTiltDeg = calculateTorsoTilt(
            activeJoints->shoulder,
            activeJoints->hip);
    }

    const bool hasUsableChain = activeJoints != nullptr &&
        (hasUpperChain(*activeJoints) || hasLowerChain(*activeJoints));

    features.valid = pose.valid &&
        features.validKeypointCount >= config_.minValidKeypoints &&
        hasUsableChain;

    if (!features.valid)
    {
        ++missingFrames_;
        if (missingFrames_ > config_.maxMissingFrames)
            reset();

        return features;
    }

    missingFrames_ = 0;

    if (hasHistory_ && previousFeatures_.activeSide == features.activeSide)
    {
        smoothFeatures(
            features,
            previousFeatures_,
            config_.smoothingFactor);
    }

    previousFeatures_ = features;
    hasHistory_ = true;

    return features;
}

void PoseFeatureExtractor::reset()
{
    hasHistory_ = false;
    previousFeatures_ = PoseFeatures();
    missingFrames_ = 0;
}

void PoseFeatureExtractor::setConfig(const PoseFeatureConfig& config)
{
    config_ = config;
    config_.minKeypointConfidence = clampf(
        config_.minKeypointConfidence, 0.0f, 1.0f);
    config_.minValidKeypoints = std::max(1, config_.minValidKeypoints);
    config_.smoothingFactor = clampf(
        config_.smoothingFactor, 0.01f, 1.0f);
    config_.maxMissingFrames = std::max(0, config_.maxMissingFrames);
    reset();
}

const PoseFeatureConfig& PoseFeatureExtractor::config() const
{
    return config_;
}
