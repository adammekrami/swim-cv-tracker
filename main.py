"""
Main entry point for the computer vision application.
"""

import argparse
import sys
from pathlib import Path

import cv2
from src.tracker import YOLOTracker


def process_video(video_path: Path, output_dir: Path, display: bool = False) -> None:
    """
    Process a video file with object tracking and save output.

    :param video_path: Path to the input video.
    :param output_dir: Directory where processed outputs will be stored.
    :param display: Whether to display frames in an OpenCV window.
    """
    if not video_path.exists():
        print(f"Error: Input video '{video_path}' not found.", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"processed_{video_path.stem}.mp4"

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Unable to open video source '{video_path}'.", file=sys.stderr)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    tracker = YOLOTracker()
    frame_count = 0

    print("Processing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Run tracking (placeholder logic via YOLOTracker)
        results = tracker.track(frame)
        annotated_frame = results[0].plot() if results else frame

        out.write(annotated_frame)

        if display:
            cv2.imshow("CV Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Processing interrupted by user.")
                break

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Finished processing {frame_count} frames.")
    print(f"Processed video saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Computer Vision YOLO Tracking Application")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (default: looks in data/raw_videos)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed videos and artifacts",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video output in an OpenCV window while processing",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw_videos"
    processed_dir = project_root / args.output_dir

    if args.video:
        target_video = Path(args.video)
    else:
        # Check if there are any MP4 files in data/raw_videos
        videos = list(raw_dir.glob("*.mp4"))
        if videos:
            target_video = videos[0]
            print(f"Found input video in data/raw_videos: {target_video.name}")
        else:
            print("No video specified and no .mp4 files found in data/raw_videos.")
            print(f"Place input videos in: {raw_dir}")
            print("Or run: python main.py --video /path/to/video.mp4")
            return

    process_video(target_video, processed_dir, display=args.display)


if __name__ == "__main__":
    main()
