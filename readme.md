# Yolo-Object-Detection-With-Intel-Realsense-Camera
The algorithm does not change the YoloV5 source code, but creates a new python file: pyrealsense2_camera.py.
## Install
### 1.Install cuda

### 2.Create YoloV5 environment && install requirements
> conda create --name yolov5 python=3.10

> conda activate yolov5

>install pytorch  

> git clone https://github.com/Jumbo-zczlbj0/YoloV5_D415.git

> cd YoloV5_D415 

> pip install -r requirements.txt 

### 3.Install Intel RealSense SDK 2.0

> Choose the version that suits you based on your computer system：https://www.intelrealsense.com/sdk-2/

> Install pyrealsense2：pip install pyrealsense2

### 4.Install other package
voice
> pip install pyttsx3

Multithreading
> pip install threading

## Start
Download and change the weight file path
> python pyrealsense2_camera.py
