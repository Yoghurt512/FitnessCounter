#pragma once

#include "BodyPose.h"
#include "PoseFeatures.h"

struct PoseFeatureConfig
{
    float minKeypointConfidence = 0.55f;

    int minValidKeypoints = 6;

    // Weight of the current frame in exponential smoothing, in (0, 1].
    // A higher value keeps fast squat/push-up motion from being flattened.
    float smoothingFactor = 0.70f;

    int maxMissingFrames = 5;
};

class PoseFeatureExtractor
{
public:
    explicit PoseFeatureExtractor(
        const PoseFeatureConfig& config = PoseFeatureConfig());

    PoseFeatures extract(const BodyPose& pose);

    void reset();

    void setConfig(const PoseFeatureConfig& config);

    const PoseFeatureConfig& config() const;

private:
    PoseFeatureConfig config_;

    bool hasHistory_ = false;

    PoseFeatures previousFeatures_;

    int missingFrames_ = 0;
};
