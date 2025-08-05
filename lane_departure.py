
import cv2
import numpy as np

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(0, height), (1280, height), (650, 400), (600, 400)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = region_of_interest(edges)
    return cropped_edges

def detect_lines(cropped_edges):
    return cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, 
                           np.array([]), minLineLength=100, maxLineGap=50)

def draw_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return cv2.addWeighted(img, 0.8, line_img, 1, 1)

def trigger_alert(frame, lines):
    left_line_detected = False
    right_line_detected = False
    width = frame.shape[1]

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if slope < -0.5:
                    left_line_detected = True
                elif slope > 0.5:
                    right_line_detected = True

    if not left_line_detected or not right_line_detected:
        cv2.putText(frame, "!! Lane Departure Detected !!", (200, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

# Main execution
video = cv2.VideoCapture("road_video2.mp4")  # Replace with 0 for webcam

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    processed = process_frame(frame)
    lines = detect_lines(processed)
    trigger_alert(frame, lines)
    result = draw_lines(frame, lines)

    cv2.imshow("Lane Departure Warning", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
