import math


class CentroidTracker:
    """
    A very simple centroid tracker for beginners.

    Idea:
    - Every detected object gets an ID.
    - In the next frame, we compare new object centers with old centers.
    - If the distance is small, we assume it is the same vehicle.
    """

    def __init__(self, distance_threshold=50, max_missing_frames=10):
        # Dictionary to store object_id -> center point (x, y)
        self.objects = {}

        # Dictionary to store how many frames an object was not seen
        self.missing_frames = {}

        # Next new ID to assign
        self.next_object_id = 0

        # Maximum allowed distance between old center and new center
        self.distance_threshold = distance_threshold

        # If object is missing for too many frames, remove it
        self.max_missing_frames = max_missing_frames

    def register(self, center):
        """Add a new object with a new ID."""
        self.objects[self.next_object_id] = center
        self.missing_frames[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove object if it disappears for many frames."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.missing_frames:
            del self.missing_frames[object_id]

    def update(self, detections):
        """
        Update tracker using current frame detections.

        Parameters:
            detections: list of center points [(x1, y1), (x2, y2), ...]

        Returns:
            dict of object_id -> center point
        """

        # If no detections found in the current frame
        if len(detections) == 0:
            # Increase missing frame count for all old objects
            for object_id in list(self.missing_frames.keys()):
                self.missing_frames[object_id] += 1

                # Remove object if missing too long
                if self.missing_frames[object_id] > self.max_missing_frames:
                    self.deregister(object_id)

            return self.objects

        # If tracker is empty, register all detections directly
        if len(self.objects) == 0:
            for center in detections:
                self.register(center)
            return self.objects

        # Convert old objects into lists for easy comparison
        object_ids = list(self.objects.keys())
        object_centers = list(self.objects.values())

        # Keep track of which detections are already matched
        used_detection_indexes = set()
        updated_objects = {}
        updated_missing_frames = {}

        # Try matching every old object with the nearest new detection
        for i, old_center in enumerate(object_centers):
            best_distance = float("inf")
            best_index = -1

            for j, new_center in enumerate(detections):
                if j in used_detection_indexes:
                    continue

                distance = math.hypot(old_center[0] - new_center[0], old_center[1] - new_center[1])

                if distance < best_distance:
                    best_distance = distance
                    best_index = j

            object_id = object_ids[i]

            # If nearest detection is close enough, update object position
            if best_index != -1 and best_distance < self.distance_threshold:
                updated_objects[object_id] = detections[best_index]
                updated_missing_frames[object_id] = 0
                used_detection_indexes.add(best_index)
            else:
                # Vehicle not found in this frame, increase missing count
                self.missing_frames[object_id] += 1
                if self.missing_frames[object_id] <= self.max_missing_frames:
                    updated_objects[object_id] = old_center
                    updated_missing_frames[object_id] = self.missing_frames[object_id]

        # Register any new detections that were not matched
        for j, center in enumerate(detections):
            if j not in used_detection_indexes:
                self.register(center)
                new_id = self.next_object_id - 1
                updated_objects[new_id] = center
                updated_missing_frames[new_id] = 0

        # Replace old tracker data with updated data
        self.objects = updated_objects
        self.missing_frames = updated_missing_frames

        return self.objects