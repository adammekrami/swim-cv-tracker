"""
YOLO Pose Estimation and Persistent Object Tracking Module.

This module implements a persistent computer vision tracking pipeline using
Ultralytics YOLOv8 Pose (yolov8n-pose.pt) and ByteTrack (bytetrack.yaml) to track
swimmers and estimate skeletal keypoints (shoulders, elbows, wrists, etc.)
across video frames while eliminating bounding box flickering and identity swaps.

Key Features:
    - Pose estimation and keypoints extraction across frames using YOLOv8-pose.
    - Persistent tracking across frames using ByteTrack Kalman filter and Hungarian matching.
    - COCO class_id 0 ('person') filtering.
    - Skeletal keypoint visualization using Ultralytics built-in plotting (results[0].plot()).
    - Raw keypoint coordinate extraction and terminal logging for data inspection.
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
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# ==============================================================================
# Configuration Constants
# ==============================================================================
COCO_PERSON_CLASS_ID = 0  # Class index for 'person' in standard COCO dataset
DEFAULT_MODEL_WEIGHTS = "yolov8n-pose.pt"  # YOLOv8 nano pose estimation pre-trained weights
DEFAULT_CONF_THRESHOLD = 0.25  # Minimum confidence threshold (optimized for swimmers)
DEFAULT_TRACKER_CONFIG = "bytetrack.yaml"  # ByteTrack algorithm configuration
TRAJECTORY_MAX_POINTS = 60  # Number of historical points to keep in trajectory tail


# ==============================================================================
# Stroke Analysis Time-Series Buffers
# ==============================================================================
left_wrist_y: List[float] = []
right_wrist_y: List[float] = []


class YOLOTracker:
    """
    Persistent Object Tracker using YOLOv8 Pose and ByteTrack.
    Maintains track history, identity persistence through occlusions,
    extracts skeletal keypoints, and renders visual skeleton overlays.
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

        :param model_weights: Pre-trained YOLO weights (e.g., 'yolov8n-pose.pt').
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

        # Frame counter for tracking & telemetry
        self.frame_count: int = 0
        self.latest_results: Optional[Any] = None
        self.latest_keypoints: Optional[np.ndarray] = None

        # Trajectory history: stores center coordinates (x, y) deque for each unique track_id
        self.track_history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.trajectory_history_len)
        )

        # Initialize YOLO model
        print(f"[YOLOTracker] Initializing YOLO pose model '{self.model_weights}'...")
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
        Extracts skeletal keypoints and prints raw coordinates to the terminal.

        :param frame: Input BGR frame (numpy array from OpenCV).
        :return: List of detection dictionaries containing:
                 - 'bbox': [x1, y1, x2, y2] bounding box coordinates
                 - 'confidence': float detection confidence (0.0 - 1.0)
                 - 'class_id': target class ID (0)
                 - 'track_id': unique persistent integer tracking ID (or None if unassigned)
                 - 'keypoints': numpy array of shape (17, 2) coordinates (if available)
        """
        self.frame_count += 1

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
        self.latest_results = results

        # ----------------------------------------------------------------------
        # Extract Keypoints Array & Print Raw Coordinates
        # ----------------------------------------------------------------------
        if results and len(results) > 0 and results[0].keypoints is not None:
            keypoints_array = results[0].keypoints.xy.cpu().numpy()
        else:
            keypoints_array = np.empty((0, 17, 2), dtype=np.float32)

        self.latest_keypoints = keypoints_array
        print(f"Frame {self.frame_count} Keypoints Raw Coordinates:\n{keypoints_array}")

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for i, box in enumerate(boxes):
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

            det_dict: Dict[str, Any] = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "track_id": track_id,
            }
            if len(keypoints_array) > i:
                det_dict["keypoints"] = keypoints_array[i]

            detections.append(det_dict)

        return detections

    def draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        results: Optional[Any] = None,
        trail_color: Tuple[int, int, int] = (0, 191, 255),  # Deep sky blue trajectory trail
    ) -> np.ndarray:
        """
        Draw visual annotations on the frame.

        Uses ultralytics built-in plotting (results[0].plot()) to automatically
        draw skeletal keypoints (shoulders, elbows, wrists) and tracking info over
        the swimmer, and overlays motion trajectory history trails.

        :param frame: BGR image frame to annotate.
        :param detections: List of detection dicts produced by `detect()`.
        :param results: Optional Ultralytics Results list/object. If None, uses self.latest_results.
        :param trail_color: BGR color tuple for motion trajectory tail.
        :return: Annotated BGR frame.
        """
        if results is None:
            results = self.latest_results

        # Use ultralytics built-in plotting to automatically draw skeletal keypoints
        # (shoulders, elbows, wrists, etc.) and bounding boxes over the swimmer
        if results and len(results) > 0:
            annotated = results[0].plot()
        else:
            annotated = frame.copy()

        h, w = frame.shape[:2]

        # Resolution-adaptive scaling for trajectory trails
        scale = max(0.5, min(w, h) / 1000.0)
        thickness = max(2, int(scale * 2.5))
        trail_thickness = max(2, int(thickness * 0.9))

        # Render motion trajectory trails for tracked subjects
        for det in detections:
            track_id = det.get("track_id")
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

        return annotated

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run detection, pose estimation, persistent tracking, and visual annotation on a frame.

        :param frame: Raw input BGR frame.
        :return: Tuple of (annotated_frame, list_of_detections).
        """
        detections = self.detect(frame)
        annotated_frame = self.draw_annotations(frame, detections, results=self.latest_results)
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
    show_preview: bool = True,
) -> None:
    """
    OpenCV Persistent Video Tracking Loop with Pose Estimation & Stroke Signal Extraction:
      1. Open video input (default: /data/raw_videos/sample.mp4).
      2. Initialize YOLOv8n-pose model with ByteTrack persistent tracking.
      3. Loop through frames.
      4. Detect and persistently track class_id 0 ('person') and estimate pose keypoints.
      5. Extract wrist keypoints (index 9: left wrist, index 10: right wrist) for stroke detection.
      6. Append wrist Y-coordinates (handling water occlusion via zero-order hold).
      7. Draw skeletal keypoints (shoulders, elbows, wrists) using results[0].plot().
      8. Render motion trajectory history to visualize swimmer path.
      9. Clean up capture and window resources.
      10. Generate and save 'stroke_signal.png' in /data/processed/ using matplotlib.

    :param video_source: Path to video file.
    :param conf_threshold: Confidence threshold for person detection.
    :param model_weights: YOLO model weights file.
    :param tracker_config: Tracker config file (default: 'bytetrack.yaml').
    :param show_preview: Whether to display interactive OpenCV window during processing.
    """
    global left_wrist_y, right_wrist_y
    left_wrist_y.clear()
    right_wrist_y.clear()

    project_root = Path(__file__).resolve().parent.parent

    # --------------------------------------------------------------------------
    # 1. Video Input Resolution
    # --------------------------------------------------------------------------
    if video_source is None:
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
    window_name = "YOLOv8 Pose + ByteTrack Swimmer Tracker (Press 'q' to exit)"
    if show_preview:
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
            # 4. Persistent Tracking & Pose Estimation
            # ------------------------------------------------------------------
            annotated_frame, detections = tracker.process_frame(frame)

            # ------------------------------------------------------------------
            # 5. Extract Wrist Keypoints for Swimming Stroke Detection
            #    In YOLOv8 pose: index 9 is left wrist, index 10 is right wrist
            # ------------------------------------------------------------------
            current_lw_y: Optional[float] = None
            current_rw_y: Optional[float] = None

            if tracker.latest_results and len(tracker.latest_results) > 0:
                results_obj = tracker.latest_results[0]
                if (
                    results_obj.keypoints is not None
                    and results_obj.keypoints.xy is not None
                    and results_obj.keypoints.xy.shape[0] > 0
                ):
                    kps_xy = results_obj.keypoints.xy[0].cpu().numpy()
                    kps_conf = (
                        results_obj.keypoints.conf[0].cpu().numpy()
                        if results_obj.keypoints.conf is not None and results_obj.keypoints.conf.shape[0] > 0
                        else None
                    )

                    # Extract left wrist (index 9)
                    if len(kps_xy) > 9:
                        lw_x, lw_val_y = float(kps_xy[9][0]), float(kps_xy[9][1])
                        lw_conf = (
                            float(kps_conf[9])
                            if (kps_conf is not None and len(kps_conf) > 9)
                            else 1.0
                        )
                        # Detected if coordinates non-zero and confidence meets threshold
                        if (lw_x != 0.0 or lw_val_y != 0.0) and lw_conf >= conf_threshold:
                            current_lw_y = lw_val_y

                    # Extract right wrist (index 10)
                    if len(kps_xy) > 10:
                        rw_x, rw_val_y = float(kps_xy[10][0]), float(kps_xy[10][1])
                        rw_conf = (
                            float(kps_conf[10])
                            if (kps_conf is not None and len(kps_conf) > 10)
                            else 1.0
                        )
                        if (rw_x != 0.0 or rw_val_y != 0.0) and rw_conf >= conf_threshold:
                            current_rw_y = rw_val_y

            # Append Y-coordinate of both wrists for every frame
            # If a wrist isn't detected in a frame due to water occlusion, append previous frame's value
            if current_lw_y is not None:
                left_wrist_y.append(current_lw_y)
            else:
                left_wrist_y.append(left_wrist_y[-1] if left_wrist_y else 0.0)

            if current_rw_y is not None:
                right_wrist_y.append(current_rw_y)
            else:
                right_wrist_y.append(right_wrist_y[-1] if right_wrist_y else 0.0)

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
            if show_preview:
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
        if show_preview:
            cv2.destroyAllWindows()
        print(f"[Done] Processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx / max(0.001, elapsed):.2f} avg FPS).")

        # ----------------------------------------------------------------------
        # 8. Plot and Save Stroke Signal Time-Series Data
        # ----------------------------------------------------------------------
        if left_wrist_y or right_wrist_y:
            print("\n[Stroke Analysis] Generating time-series plot of wrist Y-coordinates...")
            total_pts = max(len(left_wrist_y), len(right_wrist_y))
            frame_numbers = list(range(1, total_pts + 1))

            plt.figure(figsize=(12, 6))
            if left_wrist_y:
                plt.plot(
                    frame_numbers[: len(left_wrist_y)],
                    left_wrist_y,
                    label="Left Wrist Y",
                    color="tab:blue",
                    linewidth=1.8,
                )
            if right_wrist_y:
                plt.plot(
                    frame_numbers[: len(right_wrist_y)],
                    right_wrist_y,
                    label="Right Wrist Y",
                    color="tab:orange",
                    linewidth=1.8,
                )

            plt.title("Swimming Stroke Detection - Wrist Y-Coordinate Signal", fontsize=14, fontweight="bold")
            plt.xlabel("Frame Number", fontsize=12)
            plt.ylabel("Wrist Y-Coordinate", fontsize=12)
            plt.legend(loc="upper right")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            plot_output_dir = project_root / "data" / "processed"
            plot_output_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plot_output_dir / "stroke_signal.png"

            plt.savefig(str(plot_file), dpi=300)
            plt.close()
            print(f"[Stroke Analysis] Stroke signal plot saved to: {plot_file}")


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 Pose + ByteTrack Persistent Object Tracking Loop"
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
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying OpenCV video playback window",
    )

    args = parser.parse_args()

    run_tracker(
        video_source=args.video,
        conf_threshold=args.conf,
        model_weights=args.weights,
        tracker_config=args.tracker,
        show_preview=not args.no_display,
    )
