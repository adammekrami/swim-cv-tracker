"""
YOLO Person Detection and Persistent Object Tracking Module.

This module implements a persistent computer vision tracking pipeline using
Ultralytics YOLOv8 and ByteTrack (bytetrack.yaml) to eliminate bounding box
flickering and identity swaps caused by water splashing and occlusion.

Key Features:
    - Persistent tracking across frames using ByteTrack Kalman filter and Hungarian matching.
    - COCO class_id 0 ('person') filtering.
    - Bounding boxes annotated with unique persistent Tracking ID and confidence score.
    - Motion trajectory history (tracking path) maintained and visualized across frames.
    - Interactive OpenCV video stream playback with responsive window scaling.
"""

import argparse
from collections import defaultdict, deque
from pathlib import Path
import sys
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ==============================================================================
# Configuration Constants
# ==============================================================================
COCO_PERSON_CLASS_ID = 0  # Class index for 'person' in standard COCO dataset
DEFAULT_MODEL_WEIGHTS = "yolov8n.pt"  # YOLOv8 nano pre-trained weights
DEFAULT_CONF_THRESHOLD = 0.25  # Minimum confidence threshold (optimized for swimmers)
DEFAULT_TRACKER_CONFIG = "bytetrack.yaml"  # ByteTrack algorithm configuration
TRAJECTORY_MAX_POINTS = 60  # Number of historical points to keep in trajectory tail


class YOLOTracker:
    """
    Persistent Object Tracker using YOLOv8 and ByteTrack.
    Maintains track history, identity persistence through occlusions,
    and renders visual bounding boxes with tracking IDs.
    """

    def __init__(
        self,
        model_weights: str = DEFAULT_MODEL_WEIGHTS,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        tracker_config: str = DEFAULT_TRACKER_CONFIG,
        target_class_id: int = COCO_PERSON_CLASS_ID,
        trajectory_history_len: int = TRAJECTORY_MAX_POINTS,
    ) -> None:
        """
        Initialize the persistent YOLO ByteTrack tracker.

        :param model_weights: Pre-trained YOLO weights (e.g., 'yolov8n.pt').
        :param conf_threshold: Minimum detection confidence threshold (0.0 - 1.0).
        :param tracker_config: Tracking configuration file (default: 'bytetrack.yaml').
        :param target_class_id: COCO class ID to isolate (0 = person).
        :param trajectory_history_len: Max frames to retain for trajectory visualization.
        """
        self.model_weights = model_weights
        self.conf_threshold = conf_threshold
        self.tracker_config = tracker_config
        self.target_class_id = target_class_id
        self.trajectory_history_len = trajectory_history_len

        # Trajectory history: stores center coordinates (x, y) deque for each unique track_id
        self.track_history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.trajectory_history_len)
        )

        # Initialize YOLO model
        print(f"[YOLOTracker] Initializing YOLO model '{self.model_weights}'...")
        self.model: YOLO = YOLO(self.model_weights)
        print(
            f"[YOLOTracker] Loaded model. Configured with tracker='{self.tracker_config}' "
            f"(ByteTrack), persist=True, class_id={self.target_class_id} ('person')."
        )

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform persistent tracking inference on a single frame using ByteTrack.

        Uses `model.track()` with `tracker='bytetrack.yaml'` and `persist=True`
        to preserve identity state across occlusions and water splashes.

        :param frame: Input BGR frame (numpy array from OpenCV).
        :return: List of detection dictionaries containing:
                 - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                 - 'confidence': float detection confidence (0.0 - 1.0)
                 - 'class_id': target class ID (0)
                 - 'track_id': unique persistent integer tracking ID (or None if unassigned)
        """
        # Run persistent tracking inference
        # - tracker="bytetrack.yaml": uses ByteTrack Kalman filter matching
        # - persist=True: retains Kalman filter & trajectory memory from previous frames
        # - classes=[0]: strictly restricts detection to COCO class 0 ('person')
        results = self.model.track(
            source=frame,
            classes=[self.target_class_id],
            conf=self.conf_threshold,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())

            # Double-check class filtering (class 0 = person)
            if cls_id != self.target_class_id:
                continue

            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Extract unique tracking ID assigned by ByteTrack
            track_id = int(box.id[0].item()) if box.id is not None else None

            # Update trajectory coordinate history for tracked subjects
            if track_id is not None:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.track_history[track_id].append((center_x, center_y))

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "track_id": track_id,
                }
            )

        return detections

    def draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        box_color: Tuple[int, int, int] = (0, 255, 127),  # High-visibility spring green
        trail_color: Tuple[int, int, int] = (0, 191, 255),  # Deep sky blue trajectory trail
    ) -> np.ndarray:
        """
        Draw bounding boxes with persistent tracking IDs, confidence scores,
        and trajectory trails on the frame.

        :param frame: BGR image frame to annotate.
        :param detections: List of detection dicts produced by `detect()`.
        :param box_color: BGR color tuple for bounding box and text badge.
        :param trail_color: BGR color tuple for motion trajectory tail.
        :return: Annotated BGR frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Resolution-adaptive scaling for text and geometry
        scale = max(0.5, min(w, h) / 1000.0)
        thickness = max(2, int(scale * 2.5))
        font_scale = max(0.5, scale * 0.7)
        trail_thickness = max(2, int(thickness * 0.9))

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            track_id = det["track_id"]

            # Format label displaying unique Tracking ID alongside confidence score
            if track_id is not None:
                label = f"ID #{track_id} | Conf: {conf:.2f}"
            else:
                label = f"ID: Detecting... | Conf: {conf:.2f}"

            # ------------------------------------------------------------------
            # 1. Draw Trajectory Trail
            # ------------------------------------------------------------------
            if track_id is not None and len(self.track_history[track_id]) > 1:
                trail_points = np.array(
                    self.track_history[track_id], dtype=np.int32
                ).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated,
                    [trail_points],
                    isClosed=False,
                    color=trail_color,
                    thickness=trail_thickness,
                    lineType=cv2.LINE_AA,
                )

            # ------------------------------------------------------------------
            # 2. Draw Bounding Box
            # ------------------------------------------------------------------
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            # ------------------------------------------------------------------
            # 3. Draw Label Badge Background
            # ------------------------------------------------------------------
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Position label above bounding box (or inside if near frame ceiling)
            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(w, x1 + text_w + 12)

            # Solid color background badge for maximum readability
            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                box_color,
                cv2.FILLED,
            )

            # High-contrast text on badge
            text_pos = (badge_x1 + 6, badge_y2 - 6)
            cv2.putText(
                annotated,
                label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text on vibrant badge
                thickness=max(1, int(thickness / 2)),
                lineType=cv2.LINE_AA,
            )

        return annotated

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run detection, persistent tracking, and visual annotation on a frame.

        :param frame: Raw input BGR frame.
        :return: Tuple of (annotated_frame, list_of_detections).
        """
        detections = self.detect(frame)
        annotated_frame = self.draw_annotations(frame, detections)
        return annotated_frame, detections

    # Backwards compatibility alias
    def track(self, frame: np.ndarray) -> Any:
        """Direct inference wrapper with persistent ByteTrack tracking."""
        return self.model.track(
            frame,
            classes=[self.target_class_id],
            conf=self.conf_threshold,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )


# ==============================================================================
# Pipeline Runner
# ==============================================================================
def run_tracker(
    video_source: Optional[str] = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    model_weights: str = DEFAULT_MODEL_WEIGHTS,
    tracker_config: str = DEFAULT_TRACKER_CONFIG,
) -> None:
    """
    OpenCV Persistent Video Tracking Loop:
      1. Open video input (default: /data/raw_videos/sample.mp4).
      2. Initialize YOLOv8n model with ByteTrack persistent tracking.
      3. Loop through frames.
      4. Detect and persistently track class_id 0 ('person').
      5. Draw bounding box with unique Tracking ID and confidence score.
      6. Render motion trajectory history to visualize swimmer path.
      7. Display annotated stream via cv2.imshow with 'q' to quit.
      8. Clean up capture and window resources.

    :param video_source: Path to video file.
    :param conf_threshold: Confidence threshold for person detection.
    :param model_weights: YOLO model weights file.
    :param tracker_config: Tracker config file (default: 'bytetrack.yaml').
    """
    # --------------------------------------------------------------------------
    # 1. Video Input Resolution
    # --------------------------------------------------------------------------
    if video_source is None:
        project_root = Path(__file__).resolve().parent.parent
        video_path = project_root / "data" / "raw_videos" / "sample.mp4"
    else:
        video_path = Path(video_source).resolve()

    if not video_path.exists():
        print(f"[Error] Video file not found at: {video_path}", file=sys.stderr)
        print("Please ensure 'sample.mp4' exists in /data/raw_videos/ or pass --video path.", file=sys.stderr)
        return

    print(f"[Video Input] Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[Error] OpenCV failed to open video source: {video_path}", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_ms = max(1, int(1000.0 / fps))

    print(f"[Video Info] Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")

    # --------------------------------------------------------------------------
    # 2. Model & ByteTrack Initialization
    # --------------------------------------------------------------------------
    tracker = YOLOTracker(
        model_weights=model_weights,
        conf_threshold=conf_threshold,
        tracker_config=tracker_config,
    )

    # --------------------------------------------------------------------------
    # Setup OpenCV Window
    # --------------------------------------------------------------------------
    window_name = "YOLOv8 + ByteTrack Swimmer Tracker (Press 'q' to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Comfortable scale for 4K display preview
    preview_w = min(1280, width)
    preview_h = int(preview_w * (height / width))
    cv2.resizeWindow(window_name, preview_w, preview_h)

    frame_idx = 0
    start_time = time.time()

    print("[Inference Loop] Tracking started with ByteTrack... Press 'q' in the window to quit.")

    try:
        # ----------------------------------------------------------------------
        # 3. Inference Loop (Frame-by-frame)
        # ----------------------------------------------------------------------
        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()

            if not ret:
                print("\n[Inference Loop] Reached end of video stream.")
                break

            frame_idx += 1

            # ------------------------------------------------------------------
            # 4 & 5. Persistent Tracking & Annotation
            # ------------------------------------------------------------------
            annotated_frame, detections = tracker.process_frame(frame)

            # Telemetry metrics
            inference_fps = 1.0 / max(1e-5, (time.time() - loop_start))
            active_ids = [d["track_id"] for d in detections if d["track_id"] is not None]
            id_str = ", ".join(f"#{tid}" for tid in active_ids) if active_ids else "None"

            hud_info = (
                f"Frame: {frame_idx}/{total_frames} | "
                f"Tracker: ByteTrack | Active IDs: {id_str} | "
                f"FPS: {inference_fps:.1f}"
            )

            # Overlay HUD telemetry banner
            cv2.putText(
                annotated_frame,
                hud_info,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                4,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                hud_info,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            # ------------------------------------------------------------------
            # 6. Display Output & User Interaction
            # ------------------------------------------------------------------
            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q") or key == 27:
                print(f"\n[Interrupted] User pressed 'q' to quit at frame {frame_idx}.")
                break

    finally:
        # ----------------------------------------------------------------------
        # 7. Cleanup Resources
        # ----------------------------------------------------------------------
        elapsed = time.time() - start_time
        print("\n[Cleanup] Releasing video capture and destroying OpenCV windows...")
        cap.release()
        cv2.destroyAllWindows()
        print(f"[Done] Processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx / max(0.001, elapsed):.2f} avg FPS).")


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 + ByteTrack Persistent Object Tracking Loop"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (default: /data/raw_videos/sample.mp4)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=DEFAULT_TRACKER_CONFIG,
        help=f"Tracker configuration (default: {DEFAULT_TRACKER_CONFIG})",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_MODEL_WEIGHTS,
        help=f"YOLO model weights (default: {DEFAULT_MODEL_WEIGHTS})",
    )

    args = parser.parse_args()

    run_tracker(
        video_source=args.video,
        conf_threshold=args.conf,
        model_weights=args.weights,
        tracker_config=args.tracker,
    )
