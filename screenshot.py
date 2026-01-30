import subprocess
import cv2
import numpy as np

def get_frame():
    proc = subprocess.Popen(
        ["adb", "exec-out", "screencap", "-p"],
        stdout=subprocess.PIPE
    )
    img = proc.stdout.read()
    return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)


def hold_and_release(x, y, hold_ms):
    subprocess.run([
        "adb", "shell", "input", "touchscreen", "swipe",
        str(x), str(y), str(x), str(y), str(hold_ms)
    ])


frame = get_frame()
new_dimensions = (1200, 540) # (width, height)
resized_image = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
cv2.imshow("screen", resized_image)
cv2.waitKey(0)
