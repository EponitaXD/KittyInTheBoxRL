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
#new_dimensions = (1200, 540) # (width, height)
#resized_image = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)
#cv2.imshow("screen", resized_image)
#cv2.waitKey(0)

# crop frame
h, w, _ = frame.shape

# Crop area where cat + box live
crop = frame[int(0.30*h):int(0.75*h), int(0.20*w):int(0.80*w)]

# convert to gray scale and hsv
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

# Upper red range
# Strong red only (high saturation!)
lower_red1 = np.array([0, 200, 200])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 200, 200])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

red_mask = cv2.bitwise_or(mask1, mask2)

# Debuging the red classes
#vis = red_mask.copy()
#cv2.imshow("debug", vis)
#cv2.waitKey(0)

kernel = np.ones((3,3), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

# Box (dark)
_, dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

# Cat (bright)
_, bright = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

# Detect box Opening
contours, _ = cv2.findContours(
    dark,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
box_candidate = None
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w > h*2 and w*h > 2000:
        box_candidate = (x, y, w, h)


# Get the cat position
contours, _ = cv2.findContours(
    red_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cat_candidate = None
best_area = 0

for c in contours:
    area = cv2.contourArea(c)
    if area > best_area:
        best_area = area
        cat_candidate = c

if cat_candidate is not None:
    M = cv2.moments(cat_candidate)
    if M["m00"] != 0:
        cat_x = int(M["m10"] / M["m00"])
        cat_y = int(M["m01"] / M["m00"])


# Compute the State Vector
box_x, box_y, box_w, box_h = box_candidate
box_center_x = box_x + box_w / 2
#cat_x, cat_y, cat_w, cat_h = cat_candidate

distance = box_center_x - cat_x
normalized_distance = distance / crop.shape[1]
normalized_width = box_w / crop.shape[1]

state = np.array(
    [normalized_distance, normalized_width],
    dtype=np.float32
)

# Visualize
vis = crop.copy()
if box_candidate:
    x,y,w,h = box_candidate
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)

if cat_candidate is not None:
    cv2.circle(vis, (cat_x, cat_y), 5, (0,0,255), -1)

cv2.imshow("debug", vis)
cv2.waitKey(0)

