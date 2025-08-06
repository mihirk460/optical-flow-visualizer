import cv2
import numpy as np

# ---- Tunable Parameters ----
camera_index = 0
resize_width = 960
resize_height = 720
flow_params = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}
arrow_stride = 15  # int: space between arrows (pixels)
arrow_color = (0, 255, 0)  # tuple: (B, G, R) -> Green
arrow_thickness = 1  # int: arrow line thickness
arrow_tip_length = 0.3  # float: arrowhead size

# ---- Start Webcam ----
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (resize_width, resize_height))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame capture failed.")
        break

    frame = cv2.resize(frame, (resize_width, resize_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- Compute Dense Optical Flow ----
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        flow_params['pyr_scale'],
        flow_params['levels'],
        flow_params['winsize'],
        flow_params['iterations'],
        flow_params['poly_n'],
        flow_params['poly_sigma'],
        flow_params['flags']
    )

    # ---- Draw Flow Vectors ----
    # Iterate over a grid of points
    for y in range(0, flow.shape[0], arrow_stride):
        for x in range(0, flow.shape[1], arrow_stride):
            dx, dy = flow[y, x]  # flow vector at (x, y)
            start_point = (x, y)
            end_point = (int(x + dx), int(y + dy))
            cv2.arrowedLine(
                frame, start_point, end_point,
                arrow_color, arrow_thickness, tipLength=arrow_tip_length
            )

    # ---- Show Frame with Flow ----
    window_name = "Optical Flow (Arrows)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    cv2.resizeWindow(window_name, resize_width, resize_height)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()

# ---- Cleanup ----
cap.release()
cv2.destroyAllWindows()

# Notes:
# - You can further tune flow_params (e.g., 'winsize', 'levels', 'iterations') for different motion sensitivity or noise.
# - Current parameters are a good balance for general use.
# - If you see artifacts or want smoother arrows, try increasing 'winsize' or 'iterations'.