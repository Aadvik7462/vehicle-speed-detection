import os
import cv2
import numpy as np

from src.tracker import CentroidTracker
from src.utils import (
    calculate_center,
    calculate_pixel_distance,
    pixel_to_kmh,
    draw_reference_line,
    put_label,
)


# =========================
# USER SETTINGS (EDIT HERE)
# =========================
INPUT_VIDEO_PATH = "input/traffic_video.mp4"
OUTPUT_VIDEO_PATH = "output/processed_video.mp4"

# Approximate scale value for beginner project
# Example meaning: 20 pixels in video = 1 meter in real world
# You may need to change this based on your camera position and road view
PIXELS_PER_METER = 20.0

# Minimum contour size to ignore small moving noise
MIN_CONTOUR_WIDTH = 40
MIN_CONTOUR_HEIGHT = 40

# Reference line position (Y coordinate)
REFERENCE_LINE_Y = 300


def main():
    """Main function that runs the whole project."""

    # Check whether input video exists
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input video not found at '{INPUT_VIDEO_PATH}'")
        print("Please put your traffic video inside the input folder and rename it to traffic_video.mp4")
        return

    # Create output folder if it does not exist
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

    # Open the input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    # Check if video opened correctly
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # If FPS is not read properly, use a safe default value
    if fps <= 0:
        fps = 30.0

    # Create VideoWriter to save processed video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    # Background subtractor helps detect moving objects
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Initialize simple centroid tracker
    tracker = CentroidTracker(distance_threshold=60, max_missing_frames=10)

    # Store previous center points for speed calculation
    previous_positions = {}

    # Store latest bounding boxes for tracked objects
    object_boxes = {}

    print("Processing video... Press 'q' to stop preview window early.")

    while True:
        # Read one frame from the video
        ret, frame = cap.read()

        # Stop loop when video ends
        if not ret:
            break

        # Optional resize for consistent output (keep original in this project)
        processed_frame = frame.copy()

        # Apply background subtraction to detect moving regions
        fg_mask = back_sub.apply(frame)

        # Remove shadows by using threshold
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Clean noise using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        # Find contours from the cleaned foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        detection_boxes = []

        # Loop through all contours and keep only large moving objects
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore very small contours (noise, leaves, light changes, etc.)
            if w < MIN_CONTOUR_WIDTH or h < MIN_CONTOUR_HEIGHT:
                continue

            center = calculate_center(x, y, w, h)
            detections.append(center)
            detection_boxes.append((x, y, w, h, center))

        # Update tracker using current detections
        tracked_objects = tracker.update(detections)

        # Match tracker centers back to bounding boxes for drawing
        object_boxes = {}
        for object_id, center in tracked_objects.items():
            best_box = None
            best_distance = float("inf")

            for (x, y, w, h, det_center) in detection_boxes:
                distance = calculate_pixel_distance(center, det_center)
                if distance < best_distance:
                    best_distance = distance
                    best_box = (x, y, w, h)

            # Assign nearest box if close enough
            if best_box is not None and best_distance < 80:
                object_boxes[object_id] = best_box

        # Draw reference line on frame
        draw_reference_line(processed_frame, REFERENCE_LINE_Y)

        # Draw each tracked vehicle with box and speed
        for object_id, center in tracked_objects.items():
            # Default speed value
            speed_kmh = 0.0

            # If previous position exists, calculate speed
            if object_id in previous_positions:
                pixel_distance = calculate_pixel_distance(previous_positions[object_id], center)
                speed_kmh = pixel_to_kmh(pixel_distance, PIXELS_PER_METER, fps, frame_gap=1)

            # Update previous position for next frame
            previous_positions[object_id] = center

            # Draw bounding box if available
            if object_id in object_boxes:
                x, y, w, h = object_boxes[object_id]
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(processed_frame, center, 4, (0, 0, 255), -1)

                # Display vehicle ID and speed above box
                label = f"ID {object_id} | {speed_kmh:.1f} km/h"
                put_label(processed_frame, label, (x, max(y - 10, 20)))

                # Show when center crosses the reference line area
                if abs(center[1] - REFERENCE_LINE_Y) < 10:
                    put_label(processed_frame, "Crossing line", (x, y + h + 20), color=(255, 0, 0))

        # Show FPS and instructions
        put_label(processed_frame, f"FPS: {fps:.1f}", (10, 30), color=(255, 255, 255))
        put_label(processed_frame, "Press q to quit preview", (10, 60), color=(255, 255, 255))

        # Save processed frame to output video
        out.write(processed_frame)

        # Show the processed frame in a window
        cv2.imshow("Vehicle Speed Detection", processed_frame)

        # Press q on keyboard to stop preview early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video objects and close windows safely
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Processing complete.")
    print(f"Output video saved at: {OUTPUT_VIDEO_PATH}")


# Run the program only when this file is executed directly
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An unexpected error occurred:")
        print(str(e))
        print("Please check your input video and settings, then try again.")