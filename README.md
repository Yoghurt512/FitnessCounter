# FitnessCounter

人体运动姿态检测与运动计数项目。

项目目前由多个模块组成，其中：

* **A 模块：人体姿态检测**
* **B 模块：基于 A 输出的后续运动分析 / 计数**
* C、D 模块由其他组员继续开发

---

# A Module - Pose Detection

A 模块负责：

1. 接收一帧 OpenCV 图像
2. 使用 YOLO11 Pose ONNX 模型进行人体姿态检测
3. 跟踪画面中的主要运动者
4. 提取人体关键点
5. 将姿态结果以 `BodyPose` 的形式返回给其他模块

B 模块不需要修改 A 内部的姿态检测代码，只需要调用 A 提供的接口。

---

## A 模块目录结构

```text
A/
├── demo/
│   └── main.cpp
│
├── include/
│   ├── BodyPose.h
│   └── PoseDetector.h
│
├── models/
│   └── yolo11n-pose.onnx
│
└── src/
    └── PoseDetector.cpp
```

各文件作用：

```text
BodyPose.h
```

定义 A 模块输出的数据结构，包括人体关节、关键点坐标、置信度等。

```text
PoseDetector.h
```

A 模块对外接口。

```text
PoseDetector.cpp
```

YOLO Pose 模型加载、推理、关键点处理、主运动者跟踪等具体实现。

```text
yolo11n-pose.onnx
```

人体姿态检测模型。

```text
demo/main.cpp
```

仅用于测试 A 模块。

B 模块不需要使用 `demo/main.cpp`。

---

# Dependencies

## OpenCV

A 模块使用 OpenCV，并使用 OpenCV DNN 加载 ONNX 模型。

因此使用 A 模块的项目需要正确配置 OpenCV。

需要能够使用：

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
```

B 模块也需要配置 OpenCV，因为 A 的接口中使用了：

```cpp
cv::Mat
cv::Point2f
cv::Rect
```

---

## ONNX Runtime

目前 **不需要单独安装 ONNX Runtime**。

A 模块通过 OpenCV：

```cpp
cv::dnn::readNetFromONNX(...)
```

直接读取：

```text
A/models/yolo11n-pose.onnx
```

---

# B 模块如何调用 A

## 1. 获取完整 GitHub 项目

建议 B 同学直接 Clone 整个项目：

```bash
git clone <repository-url>
```

之后项目中应包含：

```text
A/
├── include/
├── src/
└── models/
```

---

# 2. 将 A 的实现文件加入 B 的 Visual Studio 项目

将：

```text
A/src/PoseDetector.cpp
```

加入 B 的 Visual Studio 项目。

Visual Studio 中可以：

```text
项目
→ 添加
→ 现有项
→ 选择 A/src/PoseDetector.cpp
```

---

# 3. 添加 A 的头文件目录

在 B 项目中：

```text
项目属性
→ C/C++
→ 常规
→ 附加包含目录
```

加入：

```text
A/include
```

之后 B 的代码就可以：

```cpp
#include "PoseDetector.h"
```

---

# 4. 配置 OpenCV

B 项目必须能够正常使用 OpenCV。

请确认下面代码能够正常编译：

```cpp
#include <opencv2/opencv.hpp>
```

并正确配置：

```text
OpenCV include
OpenCV lib
OpenCV dll
```

Visual Studio 中通常需要配置：

```text
C/C++
→ 常规
→ 附加包含目录
```

以及：

```text
链接器
→ 常规
→ 附加库目录
```

和：

```text
链接器
→ 输入
→ 附加依赖项
```

---

# 5. 初始化 PoseDetector

在程序开始时创建一个 `PoseDetector`：

```cpp
#include "PoseDetector.h"

PoseDetector detector;
```

然后加载 ONNX 模型：

```cpp
if (!detector.initialize("A/models/yolo11n-pose.onnx"))
{
    std::cerr << "Failed to load pose model." << std::endl;
    return -1;
}
```

`initialize()` 通常只需要调用一次。

---

# Important: Model Path

模型文件位于：

```text
A/models/yolo11n-pose.onnx
```

代码中的：

```cpp
"A/models/yolo11n-pose.onnx"
```

是相对于程序 **Working Directory（工作目录）** 的路径。

如果运行时出现：

```text
Failed to load pose model
```

请首先检查模型路径。

必要时可以在 Visual Studio 中检查：

```text
项目属性
→ 调试
→ 工作目录
```

---

# 6. 每一帧调用 A

假设 B 已经获得了一帧：

```cpp
cv::Mat frame;
```

直接调用：

```cpp
BodyPose pose = detector.detect(frame);
```

完整示例：

```cpp
cv::Mat frame;

// 获取图像
cap >> frame;

if (!frame.empty())
{
    BodyPose pose = detector.detect(frame);

    if (pose.valid)
    {
        // B 模块在这里使用姿态检测结果
    }
}
```

---

# A 返回的数据

A 返回：

```cpp
BodyPose
```

主要包含：

```cpp
struct BodyPose
{
    std::map<Joint, KeyPoint> joints;

    bool valid;

    cv::Rect personBox;
};
```

---

## pose.valid

使用：

```cpp
if (pose.valid)
{
    // 当前人体姿态结果有效
}
```

如果：

```cpp
pose.valid == true
```

表示当前帧检测到了足够的有效人体关键点。

---

# 关节列表

目前 A 模块向其他模块提供以下关节：

```text
Head

LeftShoulder
RightShoulder

LeftElbow
RightElbow

LeftWrist
RightWrist

LeftHip
RightHip

LeftKnee
RightKnee

LeftAnkle
RightAnkle
```

对应代码中的：

```cpp
Joint::Head

Joint::LeftShoulder
Joint::RightShoulder

Joint::LeftElbow
Joint::RightElbow

Joint::LeftWrist
Joint::RightWrist

Joint::LeftHip
Joint::RightHip

Joint::LeftKnee
Joint::RightKnee

Joint::LeftAnkle
Joint::RightAnkle
```

---

# KeyPoint 数据

每个关节对应一个：

```cpp
KeyPoint
```

包含：

```cpp
cv::Point2f position;
float confidence;
bool valid;
```

---

## position

关键点在当前图像中的像素坐标：

```cpp
kp.position.x
kp.position.y
```

---

## confidence

模型给出的关键点置信度：

```cpp
kp.confidence
```

---

## valid

表示当前关键点是否达到置信度要求：

```cpp
kp.valid
```

B 模块在使用关键点前建议始终检查：

```cpp
if (kp.valid)
{
    // 使用该关键点
}
```

---

# B 获取某个关节

例如获取左手肘：

```cpp
auto it = pose.joints.find(Joint::LeftElbow);

if (it != pose.joints.end() &&
    it->second.valid)
{
    float x = it->second.position.x;
    float y = it->second.position.y;
    float confidence = it->second.confidence;

    // B 模块继续处理
}
```

---

# 获取左右肩

```cpp
auto leftShoulder =
    pose.joints.find(Joint::LeftShoulder);

auto rightShoulder =
    pose.joints.find(Joint::RightShoulder);

if (leftShoulder != pose.joints.end() &&
    rightShoulder != pose.joints.end() &&
    leftShoulder->second.valid &&
    rightShoulder->second.valid)
{
    cv::Point2f left =
        leftShoulder->second.position;

    cv::Point2f right =
        rightShoulder->second.position;

    // B 模块可以继续计算角度、距离等
}
```

---

# 获取左侧手臂三个关键点

例如 B 需要计算手臂角度：

```cpp
auto shoulder =
    pose.joints.find(Joint::LeftShoulder);

auto elbow =
    pose.joints.find(Joint::LeftElbow);

auto wrist =
    pose.joints.find(Joint::LeftWrist);

if (shoulder != pose.joints.end() &&
    elbow != pose.joints.end() &&
    wrist != pose.joints.end() &&

    shoulder->second.valid &&
    elbow->second.valid &&
    wrist->second.valid)
{
    cv::Point2f p1 =
        shoulder->second.position;

    cv::Point2f p2 =
        elbow->second.position;

    cv::Point2f p3 =
        wrist->second.position;

    // B 模块从这里开始进行角度计算
}
```

---

# 显示 A 的检测结果

如果 B 在调试时想查看 A 检测出来的骨架，可以调用：

```cpp
detector.drawPose(frame, pose);
```

例如：

```cpp
BodyPose pose = detector.detect(frame);

detector.drawPose(frame, pose);

cv::imshow("Pose", frame);
```

`drawPose()` 主要用于调试和显示。

B 的核心逻辑不依赖 `drawPose()`。

---

# B 最小调用示例

B 如果只是希望调用 A，最简单的代码如下：

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

#include "PoseDetector.h"

int main()
{
    PoseDetector detector;

    if (!detector.initialize(
        "A/models/yolo11n-pose.onnx"))
    {
        std::cerr
            << "Failed to load pose model."
            << std::endl;

        return -1;
    }

    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        return -1;
    }

    cv::Mat frame;

    while (true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        // ==========================
        // Call Module A
        // ==========================

        BodyPose pose =
            detector.detect(frame);

        // ==========================
        // Module B
        // ==========================

        if (pose.valid)
        {
            auto leftElbow =
                pose.joints.find(
                    Joint::LeftElbow
                );

            if (leftElbow != pose.joints.end() &&
                leftElbow->second.valid)
            {
                float x =
                    leftElbow->second.position.x;

                float y =
                    leftElbow->second.position.y;

                // B logic here
            }
        }

        // Optional debug visualization
        detector.drawPose(frame, pose);

        cv::imshow(
            "FitnessCounter",
            frame
        );

        if (cv::waitKey(1) == 27)
            break;
    }

    return 0;
}
```

---

# Important Notes

### 1. 不要调用 A/demo/main.cpp

```text
A/demo/main.cpp
```

只是 A 模块的独立测试程序。

其他模块不需要使用它。

---

### 2. 不要重复创建 PoseDetector

不要每一帧都这样：

```cpp
while (...)
{
    PoseDetector detector;

    detector.initialize(...);
}
```

应该在循环开始之前：

```cpp
PoseDetector detector;
detector.initialize(...);
```

然后循环中只：

```cpp
BodyPose pose =
    detector.detect(frame);
```

---

### 3. 同一个 PoseDetector 应连续处理视频帧

A 模块内部包含主运动者的跨帧跟踪状态。

因此处理连续视频时，建议一直使用同一个：

```cpp
PoseDetector detector;
```

不要每帧重新创建。

---

### 4. 切换视频或需要重新选择人物

可以调用：

```cpp
detector.resetTracking();
```

清除之前的主运动者跟踪状态。

---

# A / B Integration Summary

B 模块最核心的调用流程只有：

```cpp
#include "PoseDetector.h"
```

↓

```cpp
PoseDetector detector;
```

↓

```cpp
detector.initialize(
    "A/models/yolo11n-pose.onnx"
);
```

↓

```cpp
BodyPose pose =
    detector.detect(frame);
```

↓

```cpp
if (pose.valid)
{
    // Use pose.joints
}
```

也就是说：

```text
cv::Mat frame
      ↓
PoseDetector::detect()
      ↓
BodyPose
      ↓
pose.joints
      ↓
Module B
```

B 模块主要使用：

```cpp
pose.valid
pose.joints
```

以及每个关键点的：

```cpp
position
confidence
valid
```

完成后续运动分析。
