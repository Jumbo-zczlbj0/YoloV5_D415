import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import pyttsx3
import threading

is_speaking = False

# Initialize the Text-to-Speech engine
engine = pyttsx3.init()

# Create a thread lock
lock = threading.Lock()

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get the depth scale of the depth sensor
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Get camera internal parameters
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Load YOLOv5 model
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
    Get 3D coordinates from 2D pixel coordinates and depth values
    """
    depth = depth * depth_scale * 1000
    x3d = (x - intr.ppx) / intr.fx * depth
    y3d = (y - intr.ppy) / intr.fy * depth
    z3d = depth
    return x3d, y3d, z3d

# Create a thread to handle the broadcast
def start_speaking_thread(text):
    thread = threading.Thread(target=speak, args=(text,))
    thread.start()


def get_median_depth_circle(depth_frame, center, radius):
    """
    Get the median depth value within the specified radius with the center point as the center of the circle.
    """
    cx, cy = center
    depth_values = []

    for x in range(max(0, cx - radius), min(cx + radius + 1, depth_frame.width)):
        for y in range(max(0, cy - radius), min(cy + radius + 1, depth_frame.height)):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                depth = depth_frame.get_distance(x, y)
                if depth > 0:  # # Ignore invalid depth values
                    depth_values.append(depth)

    if depth_values:
        return np.median(depth_values)
    else:
        return 0

try:
    while True:
        # Wait for color and depth frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # perform object detection
        results = model(color_image)
        detections = results.pred[0]

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Calculate the radius as half the width and height
            radius = min((x2 - x1) // 2, (y2 - y1) // 2)
            median_depth = get_median_depth_circle(depth_frame, (center_x, center_y), radius)
            x3d, y3d, z3d = get_3d_coord(center_x, center_y, median_depth, intrinsics)

            # Get class name
            class_name = results.names[int(cls)]

            # Only broadcast when the confidence level is greater than 0.68
            if conf > 0.68:
                # Display depth on image
                cv2.putText(color_image, f"{class_name} depth: ({z3d:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                # Show results
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)

                print(f"{class_name}: depth: ({z3d:.2f}), confidence {conf:.2f}")
                class_name = results.names[int(cls)]
                text_to_speak = f"Detected {class_name}, at depth {z3d:.2f} meters, with confidence {conf:.2f}"

                if not is_speaking:
                    # Start broadcast in new thread
                    start_speaking_thread(text_to_speak)


        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

