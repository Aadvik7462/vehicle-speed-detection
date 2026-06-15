import math
import cv2


def calculate_center(x, y, w, h):
    """Return center point of a bounding box."""
    center_x = x + w // 2
    center_y = y + h // 2
    return (center_x, center_y)


def calculate_pixel_distance(point1, point2):
    """Return Euclidean distance between two center points in pixels."""
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def pixel_to_kmh(pixel_distance, pixels_per_meter, fps, frame_gap=1):
    """
    Convert pixel movement into speed in km/h.

    Formula:
    1. pixels -> meters using pixels_per_meter
    2. time = frame_gap / fps
    3. speed = distance / time
    4. m/s -> km/h by multiplying with 3.6
    """
    if pixels_per_meter <= 0 or fps <= 0 or frame_gap <= 0:
        return 0.0

    distance_in_meters = pixel_distance / pixels_per_meter
    time_in_seconds = frame_gap / fps
    speed_mps = distance_in_meters / time_in_seconds
    speed_kmh = speed_mps * 3.6
    return speed_kmh


def draw_reference_line(frame, line_y):
    """Draw a simple horizontal reference line on the road."""
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
    cv2.putText(
        frame,
        "Reference Line",
        (10, line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def put_label(frame, text, position, color=(0, 255, 0)):
    """Draw text label on frame."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )