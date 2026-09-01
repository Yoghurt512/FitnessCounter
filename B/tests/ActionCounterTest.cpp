#include "ActionCounter.h"

#include <cassert>

PoseFeatures validFeatures(
    float elbow,
    float knee,
    float hip = 170.0f,
    float torsoTilt = 80.0f)
{
    PoseFeatures features;
    features.valid = true;
    features.elbowAngleDeg = elbow;
    features.kneeAngleDeg = knee;
    features.hipAngleDeg = hip;
    features.torsoTiltDeg = torsoTilt;
    return features;
}

int main()
{
    ActionCounter pushUp(ExerciseType::PushUp);
    assert(!pushUp.update(validFeatures(170.0f, 170.0f)));
    assert(!pushUp.update(validFeatures(90.0f, 170.0f)));
    assert(pushUp.update(validFeatures(170.0f, 170.0f)));
    assert(pushUp.count() == 1);

    // Standing arm bends must not count as push-ups.
    ActionCounter standingArms(ExerciseType::PushUp);
    assert(!standingArms.update(validFeatures(90.0f, 170.0f, 170.0f, 0.0f)));
    assert(!standingArms.update(validFeatures(170.0f, 170.0f, 170.0f, 0.0f)));
    assert(standingArms.count() == 0);

    ActionCounter squat(ExerciseType::Squat);
    assert(!squat.update(validFeatures(170.0f, 170.0f, 170.0f)));
    assert(!squat.update(validFeatures(170.0f, 125.0f, 150.0f)));
    assert(squat.update(validFeatures(170.0f, 170.0f, 170.0f)));
    assert(squat.count() == 1);

    // Simulate a fast squat whose smoothed knee angle skips the old 105-degree threshold.
    ActionCounter fastSquat(ExerciseType::Squat);
    assert(!fastSquat.update(validFeatures(170.0f, 168.0f, 170.0f)));
    assert(!fastSquat.update(validFeatures(170.0f, 140.0f, 158.0f)));
    assert(fastSquat.update(validFeatures(170.0f, 165.0f, 170.0f)));
    assert(fastSquat.count() == 1);

    // Front view: fuse both legs instead of relying on only one side.
    ActionCounter frontSquat(ExerciseType::Squat);
    PoseFeatures frontReady = validFeatures(170.0f, 170.0f, 170.0f);
    frontReady.leftKneeAngleDeg = 168.0f;
    frontReady.rightKneeAngleDeg = 172.0f;
    frontReady.leftHipAngleDeg = 170.0f;
    frontReady.rightHipAngleDeg = 170.0f;
    assert(!frontSquat.update(frontReady));
    PoseFeatures frontDown = validFeatures(170.0f, 130.0f, 150.0f);
    frontDown.leftKneeAngleDeg = 128.0f;
    frontDown.rightKneeAngleDeg = 132.0f;
    frontDown.leftHipAngleDeg = 148.0f;
    frontDown.rightHipAngleDeg = 152.0f;
    assert(!frontSquat.update(frontDown));
    assert(frontSquat.update(frontReady));
    assert(frontSquat.count() == 1);
    return 0;
}
