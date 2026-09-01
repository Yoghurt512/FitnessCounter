#include "ActionCounter.h"

#include <algorithm>
#include <cmath>

namespace
{
    bool finite(float value)
    {
        return std::isfinite(value);
    }

    float chooseAngle(float first, float second, float fallback)
    {
        if (finite(first) && finite(second))
            return (first + second) * 0.5f;
        if (finite(first))
            return first;
        if (finite(second))
            return second;
        return fallback;
    }
}

ActionCounter::ActionCounter(ExerciseType type)
    : exercise_(type)
{
}

void ActionCounter::resetRepMetrics()
{
    minimumActionAngleDeg_ = std::numeric_limits<float>::infinity();
    minimumKneeAngleDeg_ = std::numeric_limits<float>::infinity();
    minimumHipAngleDeg_ = std::numeric_limits<float>::infinity();
    minimumBodyLineAngleDeg_ = std::numeric_limits<float>::infinity();
    maximumKneeInwardScore_ = 0.0f;
}

void ActionCounter::updateLiveGuidance(
    const PoseFeatures& features,
    float actionAngle,
    float hipAngle)
{
    if (exercise_ == ExerciseType::PushUp)
    {
        if (!finite(features.torsoTiltDeg) || features.torsoTiltDeg < 45.0f)
            guidance_ = "Get your body horizontal";
        else if (finite(hipAngle) && hipAngle < 155.0f)
            guidance_ = "Keep your body straight";
        else if (phase_ == Phase::Lowered && actionAngle > 100.0f)
            guidance_ = "Lower your chest more";
        else
            guidance_ = "Good push-up posture";
        return;
    }

    if (finite(features.kneeInwardScore) && features.kneeInwardScore > 0.12f)
        guidance_ = "Keep your knees outward";
    else if (phase_ == Phase::Lowered && actionAngle > 100.0f)
        guidance_ = "Squat lower";
    else
        guidance_ = "Good squat posture";
}

void ActionCounter::setExercise(ExerciseType type)
{
    exercise_ = type;
    reset();
}

ExerciseType ActionCounter::exercise() const
{
    return exercise_;
}

bool ActionCounter::update(const PoseFeatures& features)
{
    if (!features.valid)
    {
        ++invalidFrames_;
        if (invalidFrames_ > 12)
        {
            phase_ = Phase::Ready;
            previousAngleDeg_ = std::numeric_limits<float>::quiet_NaN();
        }
        guidance_ = "Keep your full body in view";
        return false;
    }

    invalidFrames_ = 0;

    if (exercise_ == ExerciseType::Plank)
    {
        const bool straightBody = finite(features.torsoTiltDeg) &&
            features.torsoTiltDeg >= 55.0f &&
            finite(features.hipAngleDeg) && features.hipAngleDeg >= 155.0f;
        if (!straightBody)
        {
            holdStartTime_ = std::chrono::steady_clock::time_point{};
            holdSeconds_ = 0.0;
            count_ = 0;
            guidance_ = "Keep your body straight and horizontal";
            lastQualityScore_ = 0;
            return false;
        }

        if (holdStartTime_ == std::chrono::steady_clock::time_point{})
            holdStartTime_ = std::chrono::steady_clock::now();
        holdSeconds_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - holdStartTime_).count();
        count_ = static_cast<int>(holdSeconds_);
        guidance_ = features.hipAngleDeg < 165.0f
            ? "Raise your hips slightly" : "Great plank - keep holding";
        lastQualityScore_ = features.hipAngleDeg >= 165.0f ? 100 : 75;
        return false;
    }

    if (exercise_ == ExerciseType::JumpingJack)
    {
        const float spread = (finite(features.armSpreadScore) &&
            finite(features.legSpreadScore))
            ? std::min(features.armSpreadScore, features.legSpreadScore)
            : std::numeric_limits<float>::quiet_NaN();
        const bool openPose = finite(features.armSpreadScore) &&
            finite(features.legSpreadScore) &&
            features.armSpreadScore >= 1.35f && features.legSpreadScore >= 1.35f;
        const bool closedPose = finite(features.armSpreadScore) &&
            finite(features.legSpreadScore) &&
            features.armSpreadScore <= 1.15f && features.legSpreadScore <= 1.15f;
        if (openPose)
        {
            phase_ = Phase::Lowered;
            guidance_ = "Good open position - bring arms and feet together";
        }
        else if (phase_ == Phase::Lowered && closedPose)
        {
            phase_ = Phase::Ready;
            ++count_;
            guidance_ = "Good jumping jack";
            lastQualityScore_ = 100;
            return true;
        }
        else if (phase_ == Phase::Ready && closedPose &&
            finite(previousSecondaryAngleDeg_) && previousSecondaryAngleDeg_ >= 1.25f)
        {
            // A very fast open-close cycle can skip the open frame. The
            // previous spread value still proves that the arms and feet opened.
            ++count_;
            guidance_ = "Good jumping jack";
            lastQualityScore_ = 100;
            previousSecondaryAngleDeg_ = spread;
            return true;
        }
        else if (!openPose && !closedPose)
        {
            guidance_ = "Open arms and feet wider";
        }
        previousSecondaryAngleDeg_ = spread;
        return false;
    }

    const float angle = exercise_ == ExerciseType::PushUp ||
        exercise_ == ExerciseType::PullUp
        ? features.elbowAngleDeg
        : chooseAngle(features.leftKneeAngleDeg,
                      features.rightKneeAngleDeg,
                      features.kneeAngleDeg);
    const float hipAngle = exercise_ == ExerciseType::Squat
        ? chooseAngle(features.leftHipAngleDeg,
                      features.rightHipAngleDeg,
                      features.hipAngleDeg)
        : features.hipAngleDeg;

    if (!finite(angle))
    {
        guidance_ = "Keep the required joints visible";
        return false;
    }

    if (exercise_ == ExerciseType::PullUp)
    {
        const bool torsoVisible = finite(features.torsoTiltDeg);
        const bool uprightBody = finite(features.torsoTiltDeg) &&
            features.torsoTiltDeg <= 45.0f;
        const bool topPosition = uprightBody &&
            finite(features.wristShoulderHeight) &&
            features.wristShoulderHeight >= -0.05f && angle <= 125.0f;
        const bool hangingPosition = uprightBody && angle >= 150.0f;
        if (!torsoVisible || !uprightBody)
        {
            guidance_ = "Keep your body upright and joints visible";
            return false;
        }
        if (phase_ == Phase::Ready && topPosition)
        {
            phase_ = Phase::Lowered;
            guidance_ = "Good top position - lower with control";
        }
        else if (phase_ == Phase::Lowered && hangingPosition)
        {
            phase_ = Phase::Ready;
            ++count_;
            lastQualityScore_ = features.wristShoulderHeight >= 0.0f ? 100 : 75;
            guidance_ = lastQualityScore_ == 100
                ? "Good pull-up" : "Counted: pull higher next time";
            return true;
        }
        else if (!topPosition && !hangingPosition)
        {
            guidance_ = phase_ == Phase::Lowered
                ? "Return to a full hang" : "Pull until your chin reaches the bar";
        }
        previousAngleDeg_ = angle;
        previousSecondaryAngleDeg_ = features.wristShoulderHeight;
        return false;
    }

    // A push-up requires a clearly non-upright torso. This rejects standing
    // arm bends even when their elbow angle looks like a push-up.
    if (exercise_ == ExerciseType::PushUp &&
        (!finite(features.torsoTiltDeg) || features.torsoTiltDeg < 45.0f))
    {
        phase_ = Phase::Ready;
        previousAngleDeg_ = angle;
        resetRepMetrics();
        guidance_ = "Get your body horizontal";
        return false;
    }

    updateLiveGuidance(features, angle, hipAngle);

    const bool squatDownPose = exercise_ == ExerciseType::Squat &&
        finite(hipAngle) && angle <= 140.0f && hipAngle <= 165.0f;
    const bool fastDown = finite(previousAngleDeg_) &&
        angle < previousAngleDeg_ - 6.0f &&
        (exercise_ == ExerciseType::PushUp ? angle <= 125.0f
                                           : angle <= 155.0f);

    const float loweredThreshold =
        exercise_ == ExerciseType::PushUp ? 110.0f : 140.0f;
    const float readyThreshold =
        exercise_ == ExerciseType::PushUp ? 150.0f : 155.0f;

    if (phase_ == Phase::Ready &&
        ((exercise_ == ExerciseType::Squat && squatDownPose) ||
         angle <= loweredThreshold || fastDown))
    {
        phase_ = Phase::Lowered;
        resetRepMetrics();
        minimumActionAngleDeg_ = angle;
        minimumKneeAngleDeg_ = exercise_ == ExerciseType::Squat
            ? angle : std::numeric_limits<float>::infinity();
        minimumHipAngleDeg_ = exercise_ == ExerciseType::Squat
            ? hipAngle : std::numeric_limits<float>::infinity();
        minimumBodyLineAngleDeg_ = exercise_ == ExerciseType::PushUp
            ? hipAngle : std::numeric_limits<float>::infinity();
        if (finite(features.kneeInwardScore))
            maximumKneeInwardScore_ = features.kneeInwardScore;
        previousAngleDeg_ = angle;
        return false;
    }

    if (phase_ == Phase::Lowered)
    {
        minimumActionAngleDeg_ = std::min(minimumActionAngleDeg_, angle);
        if (exercise_ == ExerciseType::PushUp && finite(hipAngle))
            minimumBodyLineAngleDeg_ = std::min(minimumBodyLineAngleDeg_, hipAngle);
        if (exercise_ == ExerciseType::Squat)
        {
            minimumKneeAngleDeg_ = std::min(minimumKneeAngleDeg_, angle);
            if (finite(hipAngle))
                minimumHipAngleDeg_ = std::min(minimumHipAngleDeg_, hipAngle);
            if (finite(features.kneeInwardScore))
                maximumKneeInwardScore_ = std::max(
                    maximumKneeInwardScore_, features.kneeInwardScore);
        }

        if (angle >= readyThreshold)
        {
            const bool validPushUpRange = exercise_ != ExerciseType::PushUp ||
                minimumActionAngleDeg_ <= 125.0f;
            const bool validSquatRange = exercise_ != ExerciseType::Squat ||
                (minimumKneeAngleDeg_ <= 145.0f &&
                 minimumHipAngleDeg_ <= 170.0f);

            if (validPushUpRange && validSquatRange)
            {
                phase_ = Phase::Ready;
                ++count_;
                if (exercise_ == ExerciseType::PushUp &&
                    (minimumActionAngleDeg_ > 100.0f ||
                     (finite(minimumBodyLineAngleDeg_) &&
                      minimumBodyLineAngleDeg_ < 155.0f)))
                {
                    guidance_ = "Counted: keep your body straighter";
                    lastQualityScore_ = 75;
                }
                else if (exercise_ == ExerciseType::Squat &&
                    (minimumKneeAngleDeg_ > 100.0f ||
                     maximumKneeInwardScore_ > 0.12f))
                {
                    guidance_ = "Counted: keep your knees outward";
                    lastQualityScore_ = 75;
                }
                else
                {
                    guidance_ = "Good rep";
                    lastQualityScore_ = 100;
                }
                previousAngleDeg_ = angle;
                return true;
            }

            guidance_ = exercise_ == ExerciseType::PushUp
                ? "Not counted: lower your chest more"
                : "Not counted: squat lower";
            lastQualityScore_ = 0;
            phase_ = Phase::Ready;
        }
    }

    previousAngleDeg_ = angle;
    return false;
}

void ActionCounter::reset()
{
    phase_ = Phase::Ready;
    count_ = 0;
    invalidFrames_ = 0;
    previousAngleDeg_ = std::numeric_limits<float>::quiet_NaN();
    previousSecondaryAngleDeg_ = std::numeric_limits<float>::quiet_NaN();
    resetRepMetrics();
    guidance_ = "Move into view";
    lastQualityScore_ = 0;
    holdStartTime_ = std::chrono::steady_clock::time_point{};
    holdSeconds_ = 0.0;
}

int ActionCounter::count() const
{
    return count_;
}

const char* ActionCounter::phaseName() const
{
    return phase_ == Phase::Lowered ? "Lowered" : "Ready";
}

const std::string& ActionCounter::guidance() const
{
    return guidance_;
}

int ActionCounter::lastQualityScore() const
{
    return lastQualityScore_;
}

double ActionCounter::holdSeconds() const
{
    return holdSeconds_;
}

bool ActionCounter::isTimedHold() const
{
    return exercise_ == ExerciseType::Plank;
}
