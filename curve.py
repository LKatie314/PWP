import cv2
import numpy as np
import math

# Function to crop a frame
def crop_frame(frame, x_start, y_start, width, height):
    return frame[y_start:y_start + height, x_start:x_start + width]

# Function to calculate the angle of a line
def calculate_angle(line):
    x1, y1, x2, y2 = line
    angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle

# Function to compute the quadratic Bezier curve
def quadratic_bezier_curve(P0, P1, P2, t):
    return (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

# Function to draw a quadratic Bezier curve
def draw_bezier_curve(P0, P1, P2, image, color=(0, 0, 255), thickness=2):
    t_values = np.linspace(0, 1, 100)
    curve = np.array([quadratic_bezier_curve(P0, P1, P2, t) for t in t_values], dtype=int)
    
    for i in range(len(curve) - 1):
        cv2.line(image, tuple(curve[i]), tuple(curve[i+1]), color, thickness)

# Open a video stream (e.g., from webcam or video file)
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or replace with a video file path

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the image to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(frame_gray, 150, 200, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 4)

    # Apply Hough Line Transform
    minLineLength = 30
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 30, minLineLength, maxLineGap)

    midpointx1, midpointy1, midpointx2, midpointy2 = 0, 0, 0, 0
    midpointx3, midpointy3, midpointx4, midpointy4 = 0, 0, 0, 0

    # Check if lines are detected
    if lines is not None:
        # List to hold the longest lines
        longest_lines = []

        # Calculate the length of each line and store it with its endpoints
        line_lengths = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the length of the line
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                line_lengths.append((length, (x1, y1, x2, y2)))

        # Sort lines by length in descending order
        line_lengths.sort(reverse=True, key=lambda x: x[0])

        # Initialize an empty list to store selected longest lines
        selected_lines = []

        # Function to calculate distance between two lines
        def line_distance(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            # Compute the distance between two points (x1, y1) and (x3, y3)
            dist = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
            return dist

        # Find the 4 longest lines that are at least 50 pixels apart
        for _, (x1, y1, x2, y2) in line_lengths:
            # Check if the line is sufficiently far from the already selected lines
            if len(selected_lines) == 0:
                selected_lines.append((x1, y1, x2, y2))
            else:
                is_far_enough = True
                for selected_line in selected_lines:
                    # Check distance between current line and already selected lines
                    if line_distance(selected_line, (x1, y1, x2, y2)) < 50:
                        is_far_enough = False
                        break
                if is_far_enough:
                    selected_lines.append((x1, y1, x2, y2))
                if len(selected_lines) == 4:
                    break

        # Draw the selected longest lines
        for line in selected_lines:
            for lines in selected_lines:
                x1, y1, x2, y2 = line
                a1, b1, a2, b2 = lines
                if abs((calculate_angle(line)-calculate_angle(lines))) < 15 and (a2-x2)!=0 and (b2-y2)!=0:
                    if ((calculate_angle(line)+calculate_angle(lines))) < 0:
                        if midpointx1 == 0:
                            midpointx1 = ((x1 + a1) / 2)
                            midpointy1 = ((y1 + b1) / 2)
                            midpointx2 = ((x2 + a2) / 2)
                            midpointy2 = ((y2 + b2) / 2)
                    else:
                        print('3')
                        if midpointy3 == 0:
                            midpointx3 = ((x1 + a1) / 2)
                            midpointy3 = ((y1 + b1) / 2)
                            midpointx4 = ((x2 + a2) / 2)
                            midpointy4 = ((y2 + b2) / 2)
                cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0, 0, 255), 2)

        # Set the control points for the Bezier curve
        if abs(midpointx2 - midpointx3) < abs(midpointx2 - midpointx4):
            P0 = np.array([midpointx2, midpointy2])  # First control point
            P1 = np.array([midpointx2, midpointy3])  # Midpoint of the first line
            P2 = np.array([midpointx3, midpointy3])  # Midpoint of the second line
        else:
            P0 = np.array([midpointx2, midpointy2])  # First control point
            P1 = np.array([midpointx2, midpointy4])  # Midpoint of the first line
            P2 = np.array([midpointx4, midpointy4])  # Midpoint of the second line
        draw_bezier_curve(P0, P1, P2, frame)

        # Draw the lines
        cv2.line(frame, (int(midpointx1), int(midpointy1)), (int(midpointx2), int(midpointy2)), (0, 0, 255), 2)
        cv2.line(frame, (int(midpointx3), int(midpointy3)), (int(midpointx4), int(midpointy4)), (0, 0, 255), 2)

    # Display the output video stream with processed frame
    cv2.imshow("Video Stream with Processed Frame", frame)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
