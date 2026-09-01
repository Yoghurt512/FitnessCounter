#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "PoseDetector.h"
#include "PoseFeatureExtractor.h"
#include "ActionCounter.h"
#include "AutoExerciseClassifier.h"

namespace
{
    using Clock = std::chrono::steady_clock;
    constexpr int kSetupWidth = 1100;
    constexpr int kSetupHeight = 700;
    constexpr int kCameraPanelHeight = 122;

    enum class Screen { Setup, Countdown, Training, Rest, Finished };
    enum class Difficulty { Easy, Medium, Hard, Custom };
    enum class UiCommand { PushUp, Squat, PullUp, JumpingJack, Plank, AutoTraining,
        Easy, Medium, Hard, Custom, RepsDown, RepsUp, SetsDown, SetsUp, RestDown, RestUp,
        Start, Pause, Resume, Reset, Reselect, NewSession, Exit };

    struct Plan { int reps = 10; int sets = 3; int restSeconds = 30; };
    struct UiButton { cv::Rect rect; UiCommand command; const char* label; };
    struct UiState
    {
        Screen screen = Screen::Setup;
        Difficulty difficulty = Difficulty::Medium;
        ExerciseType selectedExercise = ExerciseType::PushUp;
        Plan customPlan;
        Plan activePlan;
        int currentSet = 1;
        int frameWidth = kSetupWidth;
        bool autoMode = false;
        bool paused = false;
        bool exitRequested = false;
        bool resetRequested = false;
        bool reselectRequested = false;
        bool startRequested = false;
        bool skipRestRequested = false;
        uint64_t sessionVersion = 0;
        uint64_t resetVersion = 0;
        uint64_t reselectVersion = 0;
        uint64_t confirmVersion = 0;
        Clock::time_point phaseStart{};
    };

    struct WorkerControl
    {
        Screen screen = Screen::Setup;
        Difficulty difficulty = Difficulty::Medium;
        ExerciseType selectedExercise = ExerciseType::PushUp;
        Plan activePlan;
        int currentSet = 1;
        bool autoMode = false;
        bool paused = false;
        uint64_t sessionVersion = 0;
        uint64_t resetVersion = 0;
        uint64_t reselectVersion = 0;
        uint64_t confirmVersion = 0;
    };

    struct InferenceResult
    {
        cv::Mat frame;
        BodyPose pose;
        ExerciseType exercise = ExerciseType::PushUp;
        int count = 0;
        double holdSeconds = 0.0;
        bool timedHold = false;
        int quality = 0;
        std::string phase = "Ready";
        std::string guidance = "Move into view";
        bool personLocked = false;
        std::string classifierStatus = "Manual mode";
        double inferenceFps = 0.0;
        uint64_t sequence = 0;
    };

    struct Pipeline
    {
        std::mutex frameMutex;
        std::condition_variable frameReady;
        cv::Mat latestFrame;
        cv::Mat latestDisplayFrame;
        uint64_t frameSequence = 0;
        std::atomic<bool> captureFinished{false};
        std::mutex resultMutex;
        InferenceResult latestResult;
        bool hasResult = false;
        std::mutex controlMutex;
        WorkerControl control;
        std::atomic<bool> stop{false};
        std::atomic<bool> modelReady{false};
        std::atomic<bool> modelFailed{false};
    };

    Plan planFor(Difficulty difficulty, const Plan& custom)
    {
        if (difficulty == Difficulty::Easy) return {8, 2, 40};
        if (difficulty == Difficulty::Medium) return {12, 3, 30};
        if (difficulty == Difficulty::Hard) return {15, 4, 20};
        return custom;
    }

    void addRow(std::vector<UiButton>& out, int width, int y,
        const std::vector<std::pair<UiCommand, const char*>>& items, int height)
    {
        const int margin = 50, gap = 14;
        const int itemWidth = (width - 2 * margin - static_cast<int>(items.size() - 1) * gap) /
            static_cast<int>(items.size());
        for (std::size_t i = 0; i < items.size(); ++i)
            out.push_back({cv::Rect(margin + static_cast<int>(i) * (itemWidth + gap), y,
                itemWidth, height), items[i].first, items[i].second});
    }

    std::vector<UiButton> makeButtons(const UiState& ui)
    {
        std::vector<UiButton> buttons;
        if (ui.screen == Screen::Setup)
        {
            addRow(buttons, kSetupWidth, 105, {{UiCommand::PushUp, "PUSH-UP"}, {UiCommand::Squat, "SQUAT"},
                {UiCommand::PullUp, "PULL-UP"}, {UiCommand::JumpingJack, "JUMPING JACK"},
                {UiCommand::Plank, "PLANK"}, {UiCommand::AutoTraining, "AUTO DETECT"}}, 54);
            addRow(buttons, kSetupWidth, 218, {{UiCommand::Easy, "EASY"}, {UiCommand::Medium, "MEDIUM"},
                {UiCommand::Hard, "HARD"}, {UiCommand::Custom, "CUSTOM"}}, 50);
            buttons.push_back({cv::Rect(50, 325, 115, 42), UiCommand::RepsDown, "REPS -"});
            buttons.push_back({cv::Rect(175, 325, 115, 42), UiCommand::RepsUp, "REPS +"});
            buttons.push_back({cv::Rect(345, 325, 115, 42), UiCommand::SetsDown, "SETS -"});
            buttons.push_back({cv::Rect(470, 325, 115, 42), UiCommand::SetsUp, "SETS +"});
            buttons.push_back({cv::Rect(640, 325, 115, 42), UiCommand::RestDown, "REST -"});
            buttons.push_back({cv::Rect(765, 325, 115, 42), UiCommand::RestUp, "REST +"});
            buttons.push_back({cv::Rect(50, 565, kSetupWidth - 100, 70), UiCommand::Start, "START WORKOUT"});
        }
        else if (ui.screen == Screen::Training)
        {
            addRow(buttons, ui.frameWidth, 18, {{ui.paused ? UiCommand::Resume : UiCommand::Pause,
                ui.paused ? "RESUME" : "PAUSE"}, {UiCommand::Reset, "RESET SET"},
                {UiCommand::Reselect, "RESELECT PERSON"}, {UiCommand::Exit, "EXIT"}}, 42);
        }
        else if (ui.screen == Screen::Rest)
        {
            addRow(buttons, ui.frameWidth, 18, {{UiCommand::Resume, "SKIP REST"},
                {UiCommand::Reset, "RESET SET"}, {UiCommand::Reselect, "RESELECT PERSON"},
                {UiCommand::Exit, "EXIT"}}, 42);
        }
        else if (ui.screen == Screen::Finished)
            addRow(buttons, kSetupWidth, 565, {{UiCommand::NewSession, "NEW WORKOUT"}, {UiCommand::Exit, "EXIT"}}, 60);
        return buttons;
    }

    void onMouse(int event, int x, int y, int, void* userdata)
    {
        if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr) return;
        auto* ui = static_cast<UiState*>(userdata);
        for (const UiButton& button : makeButtons(*ui))
        {
            if (!button.rect.contains(cv::Point(x, y))) continue;
            switch (button.command)
            {
            case UiCommand::PushUp: ui->selectedExercise = ExerciseType::PushUp; ui->autoMode = false; break;
            case UiCommand::Squat: ui->selectedExercise = ExerciseType::Squat; ui->autoMode = false; break;
            case UiCommand::PullUp: ui->selectedExercise = ExerciseType::PullUp; ui->autoMode = false; break;
            case UiCommand::JumpingJack: ui->selectedExercise = ExerciseType::JumpingJack; ui->autoMode = false; break;
            case UiCommand::Plank: ui->selectedExercise = ExerciseType::Plank; ui->autoMode = false; break;
            case UiCommand::AutoTraining: ui->autoMode = true; break;
            case UiCommand::Easy: ui->difficulty = Difficulty::Easy; break;
            case UiCommand::Medium: ui->difficulty = Difficulty::Medium; break;
            case UiCommand::Hard: ui->difficulty = Difficulty::Hard; break;
            case UiCommand::Custom: ui->difficulty = Difficulty::Custom; break;
            case UiCommand::RepsDown: ui->customPlan.reps = std::max(1, ui->customPlan.reps - 1); ui->difficulty = Difficulty::Custom; break;
            case UiCommand::RepsUp: ui->customPlan.reps = std::min(99, ui->customPlan.reps + 1); ui->difficulty = Difficulty::Custom; break;
            case UiCommand::SetsDown: ui->customPlan.sets = std::max(1, ui->customPlan.sets - 1); ui->difficulty = Difficulty::Custom; break;
            case UiCommand::SetsUp: ui->customPlan.sets = std::min(20, ui->customPlan.sets + 1); ui->difficulty = Difficulty::Custom; break;
            case UiCommand::RestDown: ui->customPlan.restSeconds = std::max(0, ui->customPlan.restSeconds - 5); ui->difficulty = Difficulty::Custom; break;
            case UiCommand::RestUp: ui->customPlan.restSeconds = std::min(300, ui->customPlan.restSeconds + 5); ui->difficulty = Difficulty::Custom; break;
            case UiCommand::Start: ui->startRequested = true; break;
            case UiCommand::Pause: ui->paused = true; break;
            case UiCommand::Resume:
                if (ui->screen == Screen::Rest) ui->skipRestRequested = true; else ui->paused = false;
                break;
            case UiCommand::Reset: ++ui->resetVersion; ui->resetRequested = true; break;
            case UiCommand::Reselect: ++ui->reselectVersion; ui->reselectRequested = true; break;
            case UiCommand::NewSession: ui->screen = Screen::Setup; ui->currentSet = 1; ui->paused = false; break;
            case UiCommand::Exit: ui->exitRequested = true; break;
            }
            break;
        }
    }

    void drawButton(cv::Mat& canvas, const UiButton& button, bool active)
    {
        const cv::Scalar fill = active ? cv::Scalar(42, 165, 132) : cv::Scalar(43, 52, 68);
        cv::rectangle(canvas, button.rect, fill, cv::FILLED);
        cv::rectangle(canvas, button.rect, active ? cv::Scalar(110, 245, 205) : cv::Scalar(91, 105, 126), 1);
        cv::putText(canvas, button.label, cv::Point(button.rect.x + 14,
            button.rect.y + button.rect.height / 2 + 7), cv::FONT_HERSHEY_SIMPLEX,
            button.rect.width > 250 ? 0.72 : 0.48, cv::Scalar(240, 246, 250), 1, cv::LINE_AA);
    }

    void drawPoseOverlay(cv::Mat& frame, const BodyPose& pose)
    {
        if (!pose.valid) return;
        const std::vector<std::pair<Joint, Joint>> links = {
            {Joint::Head, Joint::LeftShoulder}, {Joint::Head, Joint::RightShoulder},
            {Joint::LeftShoulder, Joint::LeftElbow}, {Joint::LeftElbow, Joint::LeftWrist},
            {Joint::RightShoulder, Joint::RightElbow}, {Joint::RightElbow, Joint::RightWrist},
            {Joint::LeftShoulder, Joint::LeftHip}, {Joint::RightShoulder, Joint::RightHip},
            {Joint::LeftHip, Joint::LeftKnee}, {Joint::LeftKnee, Joint::LeftAnkle},
            {Joint::RightHip, Joint::RightKnee}, {Joint::RightKnee, Joint::RightAnkle}
        };
        for (const auto& link : links)
        {
            const auto a = pose.joints.find(link.first);
            const auto b = pose.joints.find(link.second);
            if (a != pose.joints.end() && b != pose.joints.end() && a->second.valid && b->second.valid)
                cv::line(frame, a->second.position, b->second.position, cv::Scalar(80, 220, 180), 2, cv::LINE_AA);
        }
        for (const auto& item : pose.joints)
            if (item.second.valid) cv::circle(frame, item.second.position, 4, cv::Scalar(80, 235, 255), cv::FILLED, cv::LINE_AA);
    }

    void drawSetup(cv::Mat& canvas, const UiState& ui)
    {
        canvas.setTo(cv::Scalar(24, 30, 42));
        cv::putText(canvas, "FITNESS COUNTER", cv::Point(50, 48), cv::FONT_HERSHEY_SIMPLEX,
            1.1, cv::Scalar(105, 235, 200), 2, cv::LINE_AA);
        cv::putText(canvas, "Choose an exercise and build your workout", cv::Point(52, 77),
            cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(170, 184, 200), 1, cv::LINE_AA);
        cv::putText(canvas, "EXERCISE", cv::Point(50, 98), cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(125, 145, 168), 1, cv::LINE_AA);
        cv::putText(canvas, "DIFFICULTY", cv::Point(50, 207), cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(125, 145, 168), 1, cv::LINE_AA);
        cv::putText(canvas, "CUSTOM PLAN  (use +/- to adjust)", cv::Point(50, 305), cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(125, 145, 168), 1, cv::LINE_AA);
        for (const UiButton& button : makeButtons(ui))
        {
            const bool active =
                (button.command == UiCommand::AutoTraining && ui.autoMode) ||
                (button.command == UiCommand::PushUp && !ui.autoMode && ui.selectedExercise == ExerciseType::PushUp) ||
                (button.command == UiCommand::Squat && !ui.autoMode && ui.selectedExercise == ExerciseType::Squat) ||
                (button.command == UiCommand::PullUp && !ui.autoMode && ui.selectedExercise == ExerciseType::PullUp) ||
                (button.command == UiCommand::JumpingJack && !ui.autoMode && ui.selectedExercise == ExerciseType::JumpingJack) ||
                (button.command == UiCommand::Plank && !ui.autoMode && ui.selectedExercise == ExerciseType::Plank) ||
                (button.command == UiCommand::Easy && ui.difficulty == Difficulty::Easy) ||
                (button.command == UiCommand::Medium && ui.difficulty == Difficulty::Medium) ||
                (button.command == UiCommand::Hard && ui.difficulty == Difficulty::Hard) ||
                (button.command == UiCommand::Custom && ui.difficulty == Difficulty::Custom) ||
                button.command == UiCommand::Start;
            drawButton(canvas, button, active);
        }
        const Plan plan = planFor(ui.difficulty, ui.customPlan);
        std::ostringstream summary;
        summary << (ui.autoMode ? "AUTO DETECT" : ExerciseName(ui.selectedExercise)) << "   |   "
            << plan.reps << " reps x " << plan.sets << " sets   |   " << plan.restSeconds << " sec rest";
        cv::putText(canvas, summary.str(), cv::Point(50, 470), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(220, 232, 240), 1, cv::LINE_AA);
        cv::putText(canvas, "Plank uses reps as hold seconds", cv::Point(50, 505), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(135, 155, 175), 1, cv::LINE_AA);
    }

    void drawCameraPanel(cv::Mat& canvas, const UiState& ui, const InferenceResult& result,
        int countdown, int restLeft, double captureFps, double displayFps)
    {
        cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols, kCameraPanelHeight), cv::Scalar(24, 30, 42), cv::FILLED);
        if (ui.screen == Screen::Countdown)
        {
            cv::putText(canvas, "GET READY", cv::Point(22, 48), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(105, 235, 200), 2, cv::LINE_AA);
            cv::putText(canvas, std::to_string(std::max(1, countdown)), cv::Point(canvas.cols / 2 - 30, 93), cv::FONT_HERSHEY_SIMPLEX, 1.7, cv::Scalar(90, 225, 255), 3, cv::LINE_AA);
            return;
        }
        if (ui.screen == Screen::Rest)
        {
            cv::putText(canvas, "REST", cv::Point(22, 48), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(105, 235, 200), 2, cv::LINE_AA);
            cv::putText(canvas, std::to_string(restLeft) + " sec", cv::Point(22, 91), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(90, 225, 255), 2, cv::LINE_AA);
            for (const UiButton& button : makeButtons(ui)) drawButton(canvas, button, false);
            return;
        }
        if (ui.screen == Screen::Finished)
        {
            cv::putText(canvas, "WORKOUT COMPLETE", cv::Point(50, 88), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(105, 235, 200), 2, cv::LINE_AA);
            return;
        }
        for (const UiButton& button : makeButtons(ui)) drawButton(canvas, button, false);
        std::ostringstream line;
        line << ExerciseName(result.exercise) << "   SET " << ui.currentSet << "/" << ui.activePlan.sets << "   "
            << (result.timedHold ? "HOLD " + std::to_string(static_cast<int>(result.holdSeconds)) + "/" + std::to_string(ui.activePlan.reps) + " SEC" :
                "COUNT " + std::to_string(result.count) + "/" + std::to_string(ui.activePlan.reps));
        cv::putText(canvas, line.str(), cv::Point(20, 89), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(235, 240, 245), 1, cv::LINE_AA);
        cv::putText(canvas, ui.paused ? "PAUSED" : (ui.autoMode ? result.classifierStatus : "MANUAL MODE"), cv::Point(canvas.cols - 270, 89), cv::FONT_HERSHEY_SIMPLEX, 0.48, ui.paused ? cv::Scalar(90, 225, 255) : cv::Scalar(165, 180, 200), 1, cv::LINE_AA);
        std::ostringstream fps;
        fps << "capture " << std::fixed << std::setprecision(1) << captureFps
            << "  infer " << result.inferenceFps << "  display " << displayFps;
        cv::putText(canvas, fps.str(), cv::Point(canvas.cols - 270, 111), cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(150, 170, 190), 1, cv::LINE_AA);
    }

    void captureWorker(Pipeline& pipeline)
    {
        cv::VideoCapture cap;
        if (!cap.open(0, cv::CAP_MSMF) && !cap.open(0, cv::CAP_DSHOW)) cap.open(0);
        if (!cap.isOpened()) { pipeline.captureFinished = true; pipeline.stop = true; return; }
        while (!pipeline.stop)
        {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) break;
            {
                std::lock_guard<std::mutex> lock(pipeline.frameMutex);
                pipeline.latestDisplayFrame = frame.clone();
                pipeline.latestFrame = std::move(frame);
                ++pipeline.frameSequence;
            }
            pipeline.frameReady.notify_one();
        }
        cap.release();
        pipeline.captureFinished = true;
        pipeline.frameReady.notify_all();
    }

    void inferenceWorker(Pipeline& pipeline)
    {
        PoseDetector detector;
        if (!detector.initialize("A/models/yolo11n-pose.onnx")) { pipeline.modelFailed = true; pipeline.stop = true; return; }
        PoseFeatureExtractor extractor;
        ActionCounter counter(ExerciseType::PushUp);
        AutoExerciseClassifier classifier;
        pipeline.modelReady = true;
        uint64_t lastSession = 0, lastReset = 0, lastReselect = 0, lastConfirm = 0;
        uint64_t processed = 0;
        auto fpsStart = Clock::now();
        double inferenceFps = 0.0;
        while (!pipeline.stop)
        {
            cv::Mat frame;
            uint64_t sequence = 0;
            {
                std::unique_lock<std::mutex> lock(pipeline.frameMutex);
                pipeline.frameReady.wait_for(lock, std::chrono::milliseconds(30), [&] { return pipeline.stop || !pipeline.latestFrame.empty() || pipeline.captureFinished; });
                if (pipeline.stop) break;
                if (pipeline.latestFrame.empty()) { if (pipeline.captureFinished) break; else continue; }
                frame = pipeline.latestFrame.clone();
                pipeline.latestFrame.release();
                sequence = pipeline.frameSequence;
            }
            WorkerControl control;
            { std::lock_guard<std::mutex> lock(pipeline.controlMutex); control = pipeline.control; }
            if (control.sessionVersion != lastSession)
            {
                counter.setExercise(control.autoMode ? ExerciseType::PushUp : control.selectedExercise);
                classifier.reset(); lastSession = control.sessionVersion;
            }
            if (control.resetVersion != lastReset) { counter.reset(); lastReset = control.resetVersion; }
            if (control.reselectVersion != lastReselect) { detector.resetMainPerson(); lastReselect = control.reselectVersion; }
            if (control.confirmVersion != lastConfirm) { detector.confirmCurrentCandidate(); lastConfirm = control.confirmVersion; }

            InferenceResult result;
            result.frame = frame;
            result.sequence = sequence;
            result.exercise = counter.exercise();
            result.count = counter.count();
            result.timedHold = counter.isTimedHold();
            result.holdSeconds = counter.holdSeconds();
            result.quality = counter.lastQualityScore();
            result.phase = counter.phaseName();
            result.guidance = counter.guidance();
            result.personLocked = detector.isMainPersonConfirmed();
            result.classifierStatus = classifier.statusText();
            if (control.screen == Screen::Training && !control.paused)
            {
                BodyPose pose = detector.detect(result.frame);
                result.pose = pose;
                const PoseFeatures features = extractor.extract(pose);
                if (control.autoMode)
                {
                    classifier.update(features);
                    if (classifier.hasExercise() && classifier.exercise() != counter.exercise()) counter.setExercise(classifier.exercise());
                }
                counter.update(features);
                detector.drawPose(result.frame, pose);
                result.exercise = counter.exercise(); result.count = counter.count(); result.timedHold = counter.isTimedHold();
                result.holdSeconds = counter.holdSeconds(); result.quality = counter.lastQualityScore(); result.phase = counter.phaseName();
                result.guidance = counter.guidance(); result.personLocked = detector.isMainPersonConfirmed(); result.classifierStatus = classifier.statusText();
                ++processed;
                const auto elapsed = std::chrono::duration<double>(Clock::now() - fpsStart).count();
                if (elapsed >= 1.0) { inferenceFps = processed / elapsed; processed = 0; fpsStart = Clock::now(); }
            }
            result.inferenceFps = inferenceFps;
            { std::lock_guard<std::mutex> lock(pipeline.resultMutex); pipeline.latestResult = std::move(result); pipeline.hasResult = true; }
        }
    }
}

int main()
{
    UiState ui;
    Pipeline pipeline;
    std::thread captureThread(captureWorker, std::ref(pipeline));
    std::thread inferenceThread(inferenceWorker, std::ref(pipeline));

    const std::string windowName = "Fitness Counter";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setMouseCallback(windowName, onMouse, &ui);
    InferenceResult result;
    auto displayStart = Clock::now();
    int displayed = 0;
    double displayFps = 0.0;
    auto captureStart = Clock::now();
    uint64_t lastCaptureSequence = 0;
    double captureFps = 0.0;

    while (!ui.exitRequested && !pipeline.stop)
    {
        if (ui.startRequested && ui.screen == Screen::Setup)
        {
            ui.activePlan = planFor(ui.difficulty, ui.customPlan); ui.currentSet = 1;
            ui.screen = Screen::Countdown; ui.phaseStart = Clock::now(); ui.startRequested = false; ui.paused = false; ++ui.sessionVersion;
        }
        if (ui.reselectRequested)
        {
            ui.screen = Screen::Setup; ui.currentSet = 1; ui.paused = false; ui.reselectRequested = false;
        }
        if (ui.resetRequested && (ui.screen == Screen::Training || ui.screen == Screen::Rest))
        {
            ui.screen = Screen::Training; ui.paused = false; ui.resetRequested = false;
        }
        if (ui.skipRestRequested && ui.screen == Screen::Rest)
        {
            ++ui.currentSet; ui.screen = Screen::Training; ui.skipRestRequested = false; ui.phaseStart = Clock::now();
        }

        if (ui.screen == Screen::Countdown)
        {
            const int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - ui.phaseStart).count());
            if (elapsed >= 3) { ui.screen = Screen::Training; ui.phaseStart = Clock::now(); }
        }
        else if (ui.screen == Screen::Rest)
        {
            const int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - ui.phaseStart).count());
            if (elapsed >= ui.activePlan.restSeconds) { ++ui.currentSet; ++ui.resetVersion; ui.screen = Screen::Training; ui.phaseStart = Clock::now(); }
        }

        {
            std::lock_guard<std::mutex> lock(pipeline.controlMutex);
            pipeline.control.screen = ui.screen; pipeline.control.difficulty = ui.difficulty; pipeline.control.selectedExercise = ui.selectedExercise;
            pipeline.control.activePlan = ui.activePlan; pipeline.control.currentSet = ui.currentSet; pipeline.control.autoMode = ui.autoMode;
            pipeline.control.paused = ui.paused; pipeline.control.sessionVersion = ui.sessionVersion; pipeline.control.resetVersion = ui.resetVersion;
            pipeline.control.reselectVersion = ui.reselectVersion; pipeline.control.confirmVersion = ui.confirmVersion;
        }
        { std::lock_guard<std::mutex> lock(pipeline.resultMutex); if (pipeline.hasResult) result = pipeline.latestResult; }
        {
            uint64_t currentCaptureSequence = 0;
            {
                std::lock_guard<std::mutex> lock(pipeline.frameMutex);
                currentCaptureSequence = pipeline.frameSequence;
            }
            const double elapsed = std::chrono::duration<double>(Clock::now() - captureStart).count();
            if (elapsed >= 1.0)
            {
                captureFps = (currentCaptureSequence - lastCaptureSequence) / elapsed;
                lastCaptureSequence = currentCaptureSequence;
                captureStart = Clock::now();
            }
        }

        if (ui.screen == Screen::Setup)
        {
            ui.frameWidth = kSetupWidth;
            cv::Mat setup(kSetupHeight, kSetupWidth, CV_8UC3); drawSetup(setup, ui); cv::imshow(windowName, setup);
        }
        else
        {
            ui.frameWidth = result.frame.empty() ? 1280 : result.frame.cols;
            cv::Mat camera;
            {
                std::lock_guard<std::mutex> lock(pipeline.frameMutex);
                if (!pipeline.latestDisplayFrame.empty()) camera = pipeline.latestDisplayFrame.clone();
            }
            if (camera.empty()) camera = result.frame.empty() ? cv::Mat(720, 1280, CV_8UC3, cv::Scalar(30, 36, 48)).clone() : result.frame;
            drawPoseOverlay(camera, result.pose);
            cv::Mat canvas(camera.rows + kCameraPanelHeight, camera.cols, camera.type(), cv::Scalar(24, 30, 42));
            camera.copyTo(canvas(cv::Rect(0, kCameraPanelHeight, camera.cols, camera.rows)));
            int countdown = 3, restLeft = 0;
            if (ui.screen == Screen::Countdown) countdown = 3 - static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - ui.phaseStart).count());
            if (ui.screen == Screen::Rest) restLeft = std::max(0, ui.activePlan.restSeconds - static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - ui.phaseStart).count()));
            drawCameraPanel(canvas, ui, result, countdown, restLeft, captureFps, displayFps); cv::imshow(windowName, canvas);
            ++displayed; const double elapsed = std::chrono::duration<double>(Clock::now() - displayStart).count();
            if (elapsed >= 1.0) { displayFps = displayed / elapsed; displayed = 0; displayStart = Clock::now(); }
        }

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') ui.exitRequested = true;
        if (key == ' ' ) ++ui.confirmVersion;
        if (key == 'r' || key == 'R') { ++ui.reselectVersion; ui.reselectRequested = true; }
        if (key == 'c' || key == 'C') { ++ui.resetVersion; ui.resetRequested = true; }
        if (key == 'p' || key == 'P') { ui.autoMode = false; ui.selectedExercise = ExerciseType::PushUp; ui.screen = Screen::Setup; }
        if (key == 's' || key == 'S') { ui.autoMode = false; ui.selectedExercise = ExerciseType::Squat; ui.screen = Screen::Setup; }
        if (key == 'h' || key == 'H') { ui.autoMode = false; ui.selectedExercise = ExerciseType::PullUp; ui.screen = Screen::Setup; }
        if (key == 'j' || key == 'J') { ui.autoMode = false; ui.selectedExercise = ExerciseType::JumpingJack; ui.screen = Screen::Setup; }
        if (key == 'l' || key == 'L') { ui.autoMode = false; ui.selectedExercise = ExerciseType::Plank; ui.screen = Screen::Setup; }
        if (key == 'a' || key == 'A') { ui.autoMode = true; ui.screen = Screen::Setup; }
    }

    pipeline.stop = true; pipeline.frameReady.notify_all();
    if (captureThread.joinable()) captureThread.join();
    if (inferenceThread.joinable()) inferenceThread.join();
    cv::destroyAllWindows();
    return 0;
}
