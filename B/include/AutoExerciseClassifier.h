#pragma once

#include "ActionCounter.h"

#include <deque>

// Lightweight automatic selection for the two actions currently supported.
// It uses recent pose changes and needs no extra training model.
class AutoExerciseClassifier
{
public:
    struct Sample
    {
        float elbowAngleDeg;
        float kneeAngleDeg;
        float hipAngleDeg;
        float torsoTiltDeg;
        float wristShoulderHeight;
        float armSpreadScore;
        float legSpreadScore;
    };

    void update(const PoseFeatures& features);
    void reset();
    bool hasExercise() const;
    ExerciseType exercise() const;
    const char* statusText() const;

private:
    static constexpr std::size_t kWindowSize = 30;
    std::deque<Sample> samples_;
    bool hasExercise_ = false;
    ExerciseType exercise_ = ExerciseType::PushUp;
    ExerciseType pendingExercise_ = ExerciseType::PushUp;
    int stableFrames_ = 0;
};
