import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def get_frame_color_histogram(frame: np.ndarray) -> np.ndarray:
    """Computes a normalized 3D HSV color histogram to represent frame color distribution."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()


def apply_pre_cut_padding(
    labels: np.ndarray,
    majority_label: int,
    minority_label: int,
    cut_before_frames: int,
) -> np.ndarray:
    """Extends CUT segments by dropping an additional N frames immediately preceding each CUT region."""
    if cut_before_frames <= 0:
        return labels

    padded_labels = labels.copy()
    n = len(labels)

    for i in range(1, n):
        if labels[i - 1] == majority_label and labels[i] == minority_label:
            pad_start = max(0, i - cut_before_frames)
            padded_labels[pad_start:i] = minority_label

    return padded_labels


def print_timeline_map(
    labels: np.ndarray,
    fps: float,
    majority_label: int,
    bar_length: int = 60,
):
    """Displays an ASCII timeline bar and list of continuous segments for each group."""
    total_frames = len(labels)
    if total_frames == 0:
        return

    print("=" * 65)
    print(" TIMELINE SEGMENTATION MAP")
    print("=" * 65)

    ascii_bar = []
    chunk_size = total_frames / bar_length
    for i in range(bar_length):
        start_idx = int(i * chunk_size)
        end_idx = int((i + 1) * chunk_size)
        chunk = labels[start_idx:end_idx]
        if len(chunk) == 0:
            ascii_bar.append(" ")
            continue
        dominant_in_chunk = int(np.bincount(chunk).argmax())
        ascii_bar.append("█" if dominant_in_chunk == majority_label else "░")

    print(f" Timeline Bar : [{''.join(ascii_bar)}]")
    print(
        f" Legend       : █ = Kept (Group {majority_label}) | ░ = Cut (Group {1 - majority_label})\n"
    )

    segments = []
    current_label = labels[0]
    start_f = 0

    for idx in range(1, total_frames):
        if labels[idx] != current_label:
            segments.append((current_label, start_f, idx - 1))
            current_label = labels[idx]
            start_f = idx
    segments.append((current_label, start_f, total_frames - 1))

    print(f" Continuous Cluster Segments ({len(segments)} total):")
    print(" -----------------------------------------------------------")
    print(
        f" {'Group':<8} {'Status':<8} {'Time Range (s)':<22} {'Frames':<10}"
    )
    print(" -----------------------------------------------------------")

    for label, s_f, e_f in segments:
        duration_f = e_f - s_f + 1
        s_sec = s_f / fps
        e_sec = (e_f + 1) / fps
        status = "KEEP" if label == majority_label else "CUT"
        print(
            f" Group {label:<2} {status:<8} {s_sec:6.2f}s - {e_sec:6.2f}s        {duration_f:<10}"
        )

    print("=" * 65 + "\n")


def remove_minority_color_group(
    input_path: str,
    output_path: str,
    cut_start_seconds: float = 0.0,
    cut_end_seconds: float = 0.0,
    cut_before_frames: int = 0,
):
    """Clusters video frames into two color groups and saves only the larger group."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Extract original exact FPS with fallback if reading returns invalid/0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
        print(f"Warning: Could not detect valid FPS. Defaulting to {fps} FPS.")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame_idx = int(cut_start_seconds * fps)
    end_frame_idx = int((total_frames / fps - cut_end_seconds) * fps)

    start_frame_idx = max(0, start_frame_idx)
    end_frame_idx = min(total_frames, end_frame_idx)

    frames_to_process = end_frame_idx - start_frame_idx

    if frames_to_process <= 0:
        print(
            f"Error: Invalid trim values. Start: {cut_start_seconds}s, End: {cut_end_seconds}s"
        )
        print(f"Video duration: {total_frames / fps:.2f}s")
        return

    print(f"\nVideo Information:")
    print(f"  Resolution: {width}x{height} @ {fps:.3f} FPS")
    print(f"  Total frames: {total_frames} ({total_frames / fps:.2f}s)")
    print(
        f"  Trimming: {cut_start_seconds:.1f}s from start, {cut_end_seconds:.1f}s from end"
    )
    print(
        f"  Processing frames: {start_frame_idx} to {end_frame_idx} ({frames_to_process} frames)"
    )
    print(
        f"  Pre-CUT padding: {cut_before_frames} additional frame(s) removed before each CUT segment\n"
    )

    # --- Pass 1: Extract feature histograms ---
    histograms = []
    pbar_pass1 = tqdm(
        total=frames_to_process,
        unit="frame",
        desc="[1/3] Extracting features",
        ascii=False,
    )

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < start_frame_idx or frame_idx >= end_frame_idx:
            frame_idx += 1
            continue

        histograms.append(get_frame_color_histogram(frame))
        pbar_pass1.update(1)
        frame_idx += 1

    pbar_pass1.close()
    cap.release()

    if not histograms:
        print("Error: No frames found in video.")
        return

    # --- Phase 2: K-Means Clustering ---
    pbar_cluster = tqdm(
        total=1, desc="[2/3] Clustering colors  ", bar_format="{l_bar}{bar}|"
    )
    X = np.array(histograms)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    counts = np.bincount(labels)
    majority_label = int(np.argmax(counts))
    minority_label = int(np.argmin(counts))
    pbar_cluster.update(1)
    pbar_cluster.close()

    if cut_before_frames > 0:
        labels = apply_pre_cut_padding(
            labels, majority_label, minority_label, cut_before_frames
        )
        counts = np.bincount(labels)

    print(
        f"\n   Group 0: {counts[0]} frames | Group 1: {counts[1]} frames\n"
        f"   Keeping Group {majority_label} ({counts[majority_label]} frames)\n"
        f"   Cutting Group {minority_label} ({counts[minority_label]} frames)\n"
    )

    print_timeline_map(labels, fps, majority_label)

    # --- Pass 2: Write out majority group frames ---
    cap = cv2.VideoCapture(input_path)
    
    # mp4v codec writes standard MP4 streams matching the target FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pbar_pass2 = tqdm(
        total=frames_to_process,
        unit="frame",
        desc="[3/3] Exporting video   ",
        ascii=False,
    )

    frame_idx = 0
    histogram_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < start_frame_idx or frame_idx >= end_frame_idx:
            frame_idx += 1
            continue

        if labels[histogram_idx] == majority_label:
            out.write(frame)

        histogram_idx += 1
        frame_idx += 1
        pbar_pass2.update(1)

    pbar_pass2.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Filtered video saved at {fps:.3f} FPS to: {output_path}")


if __name__ == "__main__":
    input_file = input("Enter input video file path: ").strip()
    output_file = input("Enter output video file path: ").strip()

    if not output_file:
        print("Error: Output file path is required.")
    elif "." not in output_file.split("/")[-1]:
        output_file = output_file + ".mp4"
        print(f"Note: No file extension provided. Using: {output_file}")

    if not input_file:
        print("Error: Input file path is required.")
    else:
        try:
            cut_start = float(
                input("Seconds to cut from beginning (default: 0): ").strip()
                or "0"
            )
            cut_end = float(
                input("Seconds to cut from end (default: 0): ").strip() or "0"
            )
            extra_cut_before = int(
                input(
                    "Additional frames to cut BEFORE each CUT segment (default: 0): "
                ).strip()
                or "0"
            )
        except ValueError:
            print("Invalid input. Using defaults.")
            cut_start = 0.0
            cut_end = 0.0
            extra_cut_before = 0

        remove_minority_color_group(
            input_file, output_file, cut_start, cut_end, extra_cut_before
        )