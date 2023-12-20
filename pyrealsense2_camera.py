import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import pyttsx3
import threading

is_speaking = False

# 初始化Text-to-Speech引擎
engine = pyttsx3.init()

# 创建一个线程锁
lock = threading.Lock()

# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 获取深度传感器的深度比例
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 获取相机内参
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# 加载YOLOv5模型
model = torch.hub.load('./', 'custom', path='best.pt', source='local')

def speak(text):
    global is_speaking
    with lock:
        is_speaking =True
        engine.say(text)
        engine.runAndWait()
        is_speaking = False

def get_3d_coord(x, y, depth, intr):
    """
    从二维像素坐标和深度值获取三维坐标
    """
    depth = depth * depth_scale * 1000
    x3d = (x - intr.ppx) / intr.fx * depth
    y3d = (y - intr.ppy) / intr.fy * depth
    z3d = depth
    return x3d, y3d, z3d

# 创建一个线程来处理播报
def start_speaking_thread(text):
    thread = threading.Thread(target=speak, args=(text,))
    thread.start()


try:
    while True:
        # 等待彩色和深度帧
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # 进行目标检测
        results = model(color_image)
        detections = results.pred[0]

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            depth = depth_frame.get_distance(center_x, center_y)
            x3d, y3d, z3d = get_3d_coord(center_x, center_y, depth, intrinsics)

            # 获取类别名称
            class_name = results.names[int(cls)]

            # 只有当置信度大于0.9时才播报
            if conf > 0.8:
                # 在图像上显示三维坐标
                cv2.putText(color_image, f"{class_name} depth: ({z3d:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                # 显示结果
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)

                print(f"{class_name}: depth: ({z3d:.2f}), confidence {conf:.2f}")
                class_name = results.names[int(cls)]
                text_to_speak = f"Detected {class_name}, at depth {z3d:.2f} meters, with confidence {conf:.2f}"

                if not is_speaking:
                    # 在新线程中启动播报
                    start_speaking_thread(text_to_speak)


        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

