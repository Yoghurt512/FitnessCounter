#include "PoseFeatureExtractor.h"

#include <cassert>
#include <cmath>

namespace
{
    void addJoint(
        BodyPose& pose,
        Joint joint,
        float x,
        float y)
    {
        KeyPoint keypoint;
        keypoint.position = cv::Point2f(x, y);
        keypoint.confidence = 0.90f;
        keypoint.valid = true;
        pose.joints[joint] = keypoint;
    }
}

int main()
{
    BodyPose pose;
    pose.valid = true;

    addJoint(pose, Joint::LeftShoulder, 0.0f, 0.0f);
    addJoint(pose, Joint::LeftElbow, 0.0f, 1.0f);
    addJoint(pose, Joint::LeftWrist, 1.0f, 1.0f);
    addJoint(pose, Joint::LeftHip, 0.0f, 2.0f);
    addJoint(pose, Joint::LeftKnee, 0.0f, 3.0f);
    addJoint(pose, Joint::LeftAnkle, 1.0f, 3.0f);

    PoseFeatureExtractor extractor;
    const PoseFeatures features = extractor.extract(pose);

    assert(features.valid);
    assert(features.activeSide == BodySide::Left);
    assert(std::abs(features.elbowAngleDeg - 90.0f) < 0.01f);
    assert(std::abs(features.kneeAngleDeg - 90.0f) < 0.01f);
    assert(std::abs(features.torsoTiltDeg) < 0.01f);

    return 0;
}
