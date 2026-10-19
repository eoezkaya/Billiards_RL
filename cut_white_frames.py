import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def get_frame_color_histogram(frame: np.ndarray) -> np.ndarray:
    """Computes a normalized 3D HSV color histogram to represent frame color distribution."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 8 bins for Hue, 4 for Saturation, 4 for Value (128 total feature dimensions)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()


def remove_minority_color_group(input_path: str, output_path: str):
    """Clusters video frames into two color groups and saves only the larger group,

    displaying percentage progress for every phase.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Pass 1: Extract feature histograms ---
    histograms = []
    pbar_pass1 = tqdm(
        total=total_frames,
        unit="frame",
        desc="[1/3] Extracting features",
        ascii=False,
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        histograms.append(get_frame_color_histogram(frame))
        pbar_pass1.update(1)

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

    print(
        f"\n   Group 0: {counts[0]} frames | Group 1: {counts[1]} frames\n"
        f"   Keeping Group {majority_label} ({counts[majority_label]} frames)\n"
        f"   Cutting Group {minority_label} ({counts[minority_label]} frames)\n"
    )

    # --- Pass 2: Write out majority group frames ---
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pbar_pass2 = tqdm(
        total=total_frames,
        unit="frame",
        desc="[3/3] Exporting video   ",
        ascii=False,
    )
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if labels[frame_idx] == majority_label:
            out.write(frame)

        frame_idx += 1
        pbar_pass2.update(1)

    pbar_pass2.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Filtered video saved to: {output_path}")


if __name__ == "__main__":
    remove_minority_color_group(
        input_path="input.mov",
        output_path="output_both_pockets.mp4",
    )