#pragma once

#include "PoseFeatures.h"

#include <limits>
#include <string>
#include <chrono>

enum class ExerciseType
{
    PushUp,
    Squat,
    PullUp,
    JumpingJack,
    Plank
};

inline const char* ExerciseName(ExerciseType type)
{
    switch (type)
    {
    case ExerciseType::PushUp: return "Push-up";
    case ExerciseType::Squat: return "Squat";
    case ExerciseType::PullUp: return "Pull-up";
    case ExerciseType::JumpingJack: return "Jumping Jack";
    case ExerciseType::Plank: return "Plank";
    default: return "Unknown";
    }
}

class ActionCounter
{
public:
    explicit ActionCounter(ExerciseType type = ExerciseType::PushUp);

    void setExercise(ExerciseType type);
    ExerciseType exercise() const;

    // Feed one valid pose per video frame. Returns true when a repetition ends.
    bool update(const PoseFeatures& features);

    void reset();
    int count() const;
    const char* phaseName() const;

    // One short user-facing instruction for the camera preview.
    const std::string& guidance() const;
    int lastQualityScore() const;
    double holdSeconds() const;
    bool isTimedHold() const;

private:
    enum class Phase
    {
        Ready,
        Lowered
    };

    ExerciseType exercise_;
    Phase phase_ = Phase::Ready;
    int count_ = 0;
    int invalidFrames_ = 0;
    float previousAngleDeg_ = std::numeric_limits<float>::quiet_NaN();
    float minimumActionAngleDeg_ = std::numeric_limits<float>::infinity();
    float minimumKneeAngleDeg_ = std::numeric_limits<float>::infinity();
    float minimumHipAngleDeg_ = std::numeric_limits<float>::infinity();
    float minimumBodyLineAngleDeg_ = std::numeric_limits<float>::infinity();
    float maximumKneeInwardScore_ = 0.0f;
    float previousSecondaryAngleDeg_ = std::numeric_limits<float>::quiet_NaN();
    std::chrono::steady_clock::time_point holdStartTime_{};
    double holdSeconds_ = 0.0;
    std::string guidance_ = "Move into view";
    int lastQualityScore_ = 0;

    void resetRepMetrics();
    void updateLiveGuidance(
        const PoseFeatures& features,
        float actionAngle,
        float hipAngle);
};
