import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable depth streaming
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a new pair of frames (depth and color)
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Convert depth image to NumPy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert depth images to pseudo-color images for better visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Show image
        cv2.imshow('Depth Image', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()