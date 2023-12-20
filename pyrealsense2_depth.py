import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化管道
pipeline = rs.pipeline()
config = rs.config()

# 启用深度流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动管道
pipeline.start(config)

try:
    while True:
        # 等待一对新的帧（深度和彩色）
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # 将深度图像转换为NumPy数组
        depth_image = np.asanyarray(depth_frame.get_data())

        # 将深度图像转换为伪彩色图像以便更好地可视化
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示图像
        cv2.imshow('Depth Image', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()