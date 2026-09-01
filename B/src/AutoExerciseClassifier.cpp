#include "AutoExerciseClassifier.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
    bool finite(float value) { return std::isfinite(value); }

    float rangeOf(const std::deque<AutoExerciseClassifier::Sample>& samples,
                  float AutoExerciseClassifier::Sample::* member)
    {
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        for (const auto& sample : samples)
        {
            const float value = sample.*member;
            if (!finite(value)) continue;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        return finite(minimum) && finite(maximum) ? maximum - minimum : 0.0f;
    }

    template <typename Predicate>
    float ratioMatching(const std::deque<AutoExerciseClassifier::Sample>& samples,
                        Predicate predicate)
    {
        if (samples.empty()) return 0.0f;
        int matching = 0;
        for (const auto& sample : samples)
            matching += predicate(sample) ? 1 : 0;
        return static_cast<float>(matching) / samples.size();
    }
}

void AutoExerciseClassifier::update(const PoseFeatures& features)
{
    if (!features.valid) return;
    samples_.push_back({features.elbowAngleDeg, features.kneeAngleDeg,
                        features.hipAngleDeg, features.torsoTiltDeg,
                        features.wristShoulderHeight,
                        features.armSpreadScore,
                        features.legSpreadScore});
    if (samples_.size() > kWindowSize) samples_.pop_front();
    if (samples_.size() < 12) return;

    const float elbowRange = rangeOf(samples_, &Sample::elbowAngleDeg);
    const float kneeRange = rangeOf(samples_, &Sample::kneeAngleDeg);
    const float hipRange = rangeOf(samples_, &Sample::hipAngleDeg);
    const float wristRange = rangeOf(samples_, &Sample::wristShoulderHeight);
    const float armRange = rangeOf(samples_, &Sample::armSpreadScore);
    const float legRange = rangeOf(samples_, &Sample::legSpreadScore);
    const float pushScore =
        0.70f * ratioMatching(samples_, [](const Sample& s) {
            return finite(s.elbowAngleDeg) && finite(s.torsoTiltDeg) &&
                s.torsoTiltDeg >= 45.0f;
        }) + 0.30f * std::min(1.0f, elbowRange / 35.0f);
    const float squatScore =
        0.55f * ratioMatching(samples_, [](const Sample& s) {
            return finite(s.kneeAngleDeg) && finite(s.hipAngleDeg) &&
                s.torsoTiltDeg < 60.0f;
        }) + 0.25f * std::min(1.0f, kneeRange / 35.0f) +
        0.20f * std::min(1.0f, hipRange / 25.0f);

    const float pullUpScore =
        0.45f * ratioMatching(samples_, [](const Sample& s) {
            return finite(s.elbowAngleDeg) && finite(s.wristShoulderHeight);
        }) + 0.35f * std::min(1.0f, elbowRange / 45.0f) +
        0.20f * std::min(1.0f, wristRange / 0.35f);
    const float jumpingJackScore =
        0.60f * ratioMatching(samples_, [](const Sample& s) {
            return finite(s.armSpreadScore) && finite(s.legSpreadScore);
        }) + 0.20f * std::min(1.0f, armRange / 0.8f) +
        0.20f * std::min(1.0f, legRange / 0.8f);
    const float plankScore =
        0.70f * ratioMatching(samples_, [](const Sample& s) {
            return finite(s.torsoTiltDeg) && finite(s.hipAngleDeg) &&
                s.torsoTiltDeg >= 55.0f && s.hipAngleDeg >= 155.0f;
        }) + 0.30f * (1.0f - std::min(1.0f, hipRange / 20.0f));

    const float scores[] = {pushScore, squatScore, pullUpScore,
        jumpingJackScore, plankScore};
    int best = 0;
    for (int i = 1; i < 5; ++i)
        if (scores[i] > scores[best]) best = i;
    if (scores[best] < 0.62f) return;
    const ExerciseType proposal = static_cast<ExerciseType>(best);
    if (proposal == pendingExercise_) ++stableFrames_;
    else { pendingExercise_ = proposal; stableFrames_ = 1; }
    if (stableFrames_ >= 6) { exercise_ = pendingExercise_; hasExercise_ = true; }
}

void AutoExerciseClassifier::reset()
{
    samples_.clear();
    hasExercise_ = false;
    exercise_ = ExerciseType::PushUp;
    pendingExercise_ = ExerciseType::PushUp;
    stableFrames_ = 0;
}

bool AutoExerciseClassifier::hasExercise() const { return hasExercise_; }
ExerciseType AutoExerciseClassifier::exercise() const { return exercise_; }

const char* AutoExerciseClassifier::statusText() const
{
    if (!hasExercise_) return "Auto: observing movement...";
    switch (exercise_)
    {
    case ExerciseType::PushUp: return "Auto: push-up";
    case ExerciseType::Squat: return "Auto: squat";
    case ExerciseType::PullUp: return "Auto: pull-up";
    case ExerciseType::JumpingJack: return "Auto: jumping jack";
    case ExerciseType::Plank: return "Auto: plank";
    default: return "Auto: observing movement...";
    }
}
