#pragma once

#include <limits>

enum class BodySide
{
    None,
    Left,
    Right
};

inline const char* BodySideName(BodySide side)
{
    switch (side)
    {
    case BodySide::Left:
        return "Left";
    case BodySide::Right:
        return "Right";
    default:
        return "None";
    }
}

// Output contract from module B to the action-recognition module.
struct PoseFeatures
{
    bool valid = false;

    BodySide activeSide = BodySide::None;

    int validKeypointCount = 0;

    float meanKeypointConfidence = 0.0f;

    float sideConfidence = 0.0f;

    float elbowAngleDeg = std::numeric_limits<float>::quiet_NaN();

    float shoulderAngleDeg = std::numeric_limits<float>::quiet_NaN();

    float hipAngleDeg = std::numeric_limits<float>::quiet_NaN();

    float kneeAngleDeg = std::numeric_limits<float>::quiet_NaN();

    // Keep both sides so action recognition can work from front or side views.
    float leftHipAngleDeg = std::numeric_limits<float>::quiet_NaN();
    float rightHipAngleDeg = std::numeric_limits<float>::quiet_NaN();
    float leftKneeAngleDeg = std::numeric_limits<float>::quiet_NaN();
    float rightKneeAngleDeg = std::numeric_limits<float>::quiet_NaN();

    // Absolute angle from vertical, where zero means an upright torso.
    float torsoTiltDeg = std::numeric_limits<float>::quiet_NaN();

    // Front-view estimate of knees moving inward. Zero is neutral; higher
    // values indicate that both knees move toward the body center.
    float kneeInwardScore = std::numeric_limits<float>::quiet_NaN();

    // Normalized geometry used by the extended actions.
    float wristShoulderHeight = std::numeric_limits<float>::quiet_NaN();
    float armSpreadScore = std::numeric_limits<float>::quiet_NaN();
    float legSpreadScore = std::numeric_limits<float>::quiet_NaN();
};
