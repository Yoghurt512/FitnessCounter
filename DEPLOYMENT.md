# Visual Studio 2022 deployment

The project is configured for the OpenCV 4.12.0 installation at:

`D:\common models\opencv460`

Open `build-vs\FitnessCounter.sln` with Visual Studio 2022. Select `x64`
and `Release` or `Debug`, then make `FitnessCounter` the startup project.
Press `F5` to run the camera pose-detection demo. Press `Esc` in the video
window to exit.

The model is included at `A\models\yolo11n-pose.onnx`; no separate ONNX
Runtime installation is required.
