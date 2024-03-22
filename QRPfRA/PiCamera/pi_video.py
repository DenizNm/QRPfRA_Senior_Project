import cv2
import numpy as np
from picamera2 import Picamera2
import time


# Initialize Picamera2
camera = Picamera2()
camera_config = camera.create_preview_configuration()
# camera_config["main"]["size"] = (1280, 720)  # Set resolution to 1280x720
# Set the frame rate if the API supports it; this is just an example



camera.configure(camera_config)

# Start the camera
camera.start()

# Allow the camera to warm up
time.sleep(2)

# Capture frames from the camera
while True:
    # Capture the frame
    frame = camera.capture_array()
    
    # Convert to a format OpenCV can use
    image = np.array(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
camera.stop()
