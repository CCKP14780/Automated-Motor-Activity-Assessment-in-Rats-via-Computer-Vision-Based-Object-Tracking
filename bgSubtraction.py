import cv2
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

VIDEO_NAME = "F8"

VIDEO_PATH = (
    rf"sleap-labeling\datasets\train\{VIDEO_NAME}.mov"
)

# Frames sampled throughout the video to estimate the background
BACKGROUND_SAMPLES = 80


# ============================================================
# Export configuration
# ============================================================

# Choose which type of video to export when pressing E.
#
# Available:
#   "original"
#   "background"
#   "difference"
#   "raw_mask"
#   "clean_mask"
#   "rat_only"
#
# Use "none" if you do not want to export.
EXPORT_VIDEO_TYPE = "rat_only"


# Output directory
EXPORT_DIR = Path(
    r"C:\Users\ICT68\Documents\GitHub"
    r"\Automated-Motor-Activity-Assessment-in-Rats-via-Computer-Vision-Based-Object-Tracking"
    r"\sleap-labeling\datasets\train"
)


# Export resolution
EXPORT_WIDTH = 1920
EXPORT_HEIGHT = 1080


# ============================================================
# Automatic threshold configuration
# ============================================================

MIN_AUTO_THRESHOLD = 3
MAX_AUTO_THRESHOLD = 50

# Higher values react faster;
# lower values are more stable.
THRESHOLD_SMOOTHING = 0.10


# ============================================================
# Image processing
# ============================================================

INITIAL_MIN_AREA = 100

BLUR_KERNEL_SIZE = 3

CLOSE_KERNEL_SIZE = 5
CLOSE_ITERATIONS = 1


# ============================================================
# Split-screen display
# ============================================================

PANEL_WIDTH = 360
PANEL_HEIGHT = 220

WINDOW_NAME = (
    "Automatic Rat Background Subtraction"
)


# ============================================================
# Display functions
# ============================================================

def resize_to_panel(
    image,
    is_mask=False
):
    """Resize while preserving aspect ratio and add black padding."""

    height, width = image.shape[:2]

    scale = min(
        PANEL_WIDTH / width,
        PANEL_HEIGHT / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    if is_mask:

        interpolation = cv2.INTER_NEAREST

    else:

        interpolation = (
            cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_LINEAR
        )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    if resized.ndim == 2:

        resized = cv2.cvtColor(
            resized,
            cv2.COLOR_GRAY2BGR
        )

    panel = np.zeros(
        (
            PANEL_HEIGHT,
            PANEL_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    x = (
        PANEL_WIDTH - new_width
    ) // 2

    y = (
        PANEL_HEIGHT - new_height
    ) // 2

    panel[
        y:y + new_height,
        x:x + new_width
    ] = resized

    return panel


def add_label(
    image,
    text
):
    """Add a label to a display panel."""

    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (PANEL_WIDTH, 38),
        (0, 0, 0),
        thickness=-1
    )

    cv2.putText(
        output,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return output


# ============================================================
# Background estimation
# ============================================================

def estimate_background(
    video_path,
    sample_count
):
    """
    Estimate the static background using the temporal median
    of frames sampled evenly throughout the video.
    """

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        capture.release()

        raise RuntimeError(
            "Could not determine the "
            "video frame count."
        )

    sample_count = min(
        sample_count,
        total_frames
    )

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        sample_count,
        dtype=np.int64
    )

    sampled_frames = []

    print(
        f"Estimating background from "
        f"{sample_count} frames..."
    )

    for index, frame_number in enumerate(
        frame_indices,
        start=1
    ):

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_number)
        )

        success, frame = (
            capture.read()
        )

        if success:

            sampled_frames.append(
                frame
            )

        print(
            f"\rSampling "
            f"{index}/{sample_count}",
            end=""
        )

    capture.release()

    print()

    if not sampled_frames:

        raise RuntimeError(
            "No frames could be sampled."
        )

    print(
        "Calculating temporal median..."
    )

    frame_stack = np.stack(
        sampled_frames,
        axis=0
    )

    background = np.median(
        frame_stack,
        axis=0
    ).astype(np.uint8)

    del frame_stack
    del sampled_frames

    print(
        "Background estimation complete."
    )

    return background


# ============================================================
# Foreground processing
# ============================================================

def calculate_difference(
    frame,
    blurred_background
):
    """Calculate the difference between a frame and the background."""

    blurred_frame = cv2.GaussianBlur(
        frame,
        (
            BLUR_KERNEL_SIZE,
            BLUR_KERNEL_SIZE
        ),
        sigmaX=0
    )

    color_difference = cv2.absdiff(
        blurred_frame,
        blurred_background
    )

    # Use the largest difference among B, G, and R.
    difference_gray = np.max(
        color_difference,
        axis=2
    ).astype(np.uint8)

    return difference_gray


def calculate_otsu_threshold(
    difference_gray
):
    """Calculate an automatic threshold using Otsu's method."""

    otsu_threshold, _ = cv2.threshold(
        difference_gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    otsu_threshold = np.clip(
        otsu_threshold,
        MIN_AUTO_THRESHOLD,
        MAX_AUTO_THRESHOLD
    )

    return float(
        otsu_threshold
    )


def create_clean_mask(
    difference_gray,
    threshold_value,
    minimum_area
):
    """Create and clean the foreground mask."""

    _, raw_mask = cv2.threshold(
        difference_gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Fill small holes and connect nearby
    # foreground regions.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            CLOSE_KERNEL_SIZE,
            CLOSE_KERNEL_SIZE
        )
    )

    connected_mask = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=CLOSE_ITERATIONS
    )

    # Find connected foreground objects.
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            connected_mask,
            connectivity=8
        )
    )

    clean_mask = np.zeros_like(
        raw_mask
    )

    # Retain every component larger
    # than minimum_area.
    for label in range(
        1,
        count
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_area:

            clean_mask[
                labels == label
            ] = 255

    return raw_mask, clean_mask


# ============================================================
# Export functions
# ============================================================

def get_export_frame(
    video_type,
    frame,
    difference_gray,
    raw_mask,
    clean_mask,
    rat_only,
    background
):
    """
    Select the frame that should be exported.
    """

    if video_type == "original":

        output_frame = frame

    elif video_type == "background":

        output_frame = background

    elif video_type == "difference":

        output_frame = cv2.applyColorMap(
            difference_gray,
            cv2.COLORMAP_TURBO
        )

    elif video_type == "raw_mask":

        output_frame = cv2.cvtColor(
            raw_mask,
            cv2.COLOR_GRAY2BGR
        )

    elif video_type == "clean_mask":

        output_frame = cv2.cvtColor(
            clean_mask,
            cv2.COLOR_GRAY2BGR
        )

    elif video_type == "rat_only":

        output_frame = rat_only

    else:

        raise ValueError(
            f"Unknown video type: "
            f"{video_type}"
        )

    return output_frame


def export_video(
    video_type,
    minimum_area,
    threshold_offset
):
    """
    Export a selected version of the video.

    The current slider values are used for the export.

    The output resolution is always 1920x1080.
    """

    valid_types = {
        "original",
        "background",
        "difference",
        "raw_mask",
        "clean_mask",
        "rat_only"
    }

    # --------------------------------------------------------
    # Export disabled
    # --------------------------------------------------------

    if video_type == "none":

        print()
        print(
            "Export skipped "
            "(EXPORT_VIDEO_TYPE = 'none')."
        )

        return

    # --------------------------------------------------------
    # Validate export type
    # --------------------------------------------------------

    if video_type not in valid_types:

        raise ValueError(
            f"Unknown video type: "
            f"{video_type}\n"
            f"Available types: "
            f"{sorted(valid_types)}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        EXPORT_DIR
        / f"{VIDEO_NAME}_{video_type}.mp4"
    )

    print()
    print("=" * 60)
    print("EXPORTING VIDEO")
    print("=" * 60)
    print(
        f"Type:       {video_type}"
    )
    print(
        f"Min area:   {minimum_area}"
    )
    print(
        f"Threshold:  {threshold_offset:+d}"
    )
    print(
        f"Resolution: "
        f"{EXPORT_WIDTH} x "
        f"{EXPORT_HEIGHT}"
    )
    print(
        f"Output:     {output_path}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Open source video
    # --------------------------------------------------------

    export_capture = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not export_capture.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    fps = export_capture.get(
        cv2.CAP_PROP_FPS
    )

    if (
        not fps
        or fps <= 0
        or not np.isfinite(fps)
    ):

        fps = 30.0

    # --------------------------------------------------------
    # Create writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            EXPORT_WIDTH,
            EXPORT_HEIGHT
        ),
        True
    )

    if not writer.isOpened():

        export_capture.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    frame_count = 0

    # --------------------------------------------------------
    # Process every frame
    # --------------------------------------------------------

    while True:

        success, frame = (
            export_capture.read()
        )

        if not success:
            break

        # Calculate difference
        difference_gray = (
            calculate_difference(
                frame,
                blurred_background
            )
        )

        # Otsu threshold
        otsu_threshold = (
            calculate_otsu_threshold(
                difference_gray
            )
        )

        # Apply CURRENT slider offset
        final_threshold = int(
            round(
                otsu_threshold
                + threshold_offset
            )
        )

        final_threshold = int(
            np.clip(
                final_threshold,
                1,
                255
            )
        )

        # Create masks
        raw_mask, clean_mask = (
            create_clean_mask(
                difference_gray,
                final_threshold,
                minimum_area
            )
        )

        # Create rat-only image
        rat_only = cv2.bitwise_and(
            frame,
            frame,
            mask=clean_mask
        )

        # Select output
        output_frame = (
            get_export_frame(
                video_type,
                frame,
                difference_gray,
                raw_mask,
                clean_mask,
                rat_only,
                estimated_background
            )
        )

        # Resize to 1920x1080
        output_frame = cv2.resize(
            output_frame,
            (
                EXPORT_WIDTH,
                EXPORT_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        writer.write(
            output_frame
        )

        frame_count += 1

        if frame_count % 100 == 0:

            print(
                f"Processed "
                f"{frame_count} frames...",
                end="\r"
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    export_capture.release()
    writer.release()

    print()
    print()
    print(
        "Export complete!"
    )
    print(
        f"Frames:     {frame_count}"
    )
    print(
        f"Resolution: "
        f"{EXPORT_WIDTH} x "
        f"{EXPORT_HEIGHT}"
    )
    print(
        f"Min area:   {minimum_area}"
    )
    print(
        f"Threshold:  "
        f"{threshold_offset:+d}"
    )
    print(
        f"Saved to:   {output_path}"
    )
    print("=" * 60)


# ============================================================
# Estimate background
# ============================================================

estimated_background = (
    estimate_background(
        VIDEO_PATH,
        BACKGROUND_SAMPLES
    )
)

blurred_background = cv2.GaussianBlur(
    estimated_background,
    (
        BLUR_KERNEL_SIZE,
        BLUR_KERNEL_SIZE
    ),
    sigmaX=0
)


# ============================================================
# Open video
# ============================================================

capture = cv2.VideoCapture(
    VIDEO_PATH
)

if not capture.isOpened():

    raise RuntimeError(
        f"Could not open video: "
        f"{VIDEO_PATH}"
    )

fps = capture.get(
    cv2.CAP_PROP_FPS
)

if (
    not fps
    or fps <= 0
    or not np.isfinite(fps)
):

    fps = 30.0

frame_delay = max(
    1,
    int(round(1000 / fps))
)


# ============================================================
# Create display and sliders
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    PANEL_WIDTH * 3,
    PANEL_HEIGHT * 2
)


# Minimum connected-component area
cv2.createTrackbar(
    "Min area",
    WINDOW_NAME,
    INITIAL_MIN_AREA,
    10000,
    lambda value: None
)


# Threshold offset:
# Slider 0–20
# Actual offset -10 to +10
INITIAL_OFFSET_POSITION = (
    0 + 10
)

cv2.createTrackbar(
    "Threshold offset",
    WINDOW_NAME,
    INITIAL_OFFSET_POSITION,
    20,
    lambda value: None
)


# ============================================================
# Read first frame
# ============================================================

success, current_frame = (
    capture.read()
)

if not success:

    capture.release()
    cv2.destroyAllWindows()

    raise RuntimeError(
        "The video contains no readable frames."
    )


paused = False
smoothed_threshold = None


# ============================================================
# Main loop
# ============================================================

while True:

    # --------------------------------------------------------
    # Get CURRENT slider values
    # --------------------------------------------------------

    minimum_area = cv2.getTrackbarPos(
        "Min area",
        WINDOW_NAME
    )

    # Slider 0–20 → offset -10 to +10
    threshold_offset = (
        cv2.getTrackbarPos(
            "Threshold offset",
            WINDOW_NAME
        )
        - 10
    )

    # --------------------------------------------------------
    # Calculate frame-to-background difference
    # --------------------------------------------------------

    difference_gray = (
        calculate_difference(
            current_frame,
            blurred_background
        )
    )

    # --------------------------------------------------------
    # Calculate automatic threshold
    # --------------------------------------------------------

    otsu_threshold = (
        calculate_otsu_threshold(
            difference_gray
        )
    )

    # --------------------------------------------------------
    # Smooth threshold changes
    # --------------------------------------------------------

    if smoothed_threshold is None:

        smoothed_threshold = (
            otsu_threshold
        )

    elif not paused:

        smoothed_threshold = (
            (1.0 - THRESHOLD_SMOOTHING)
            * smoothed_threshold
            +
            THRESHOLD_SMOOTHING
            * otsu_threshold
        )

    # --------------------------------------------------------
    # Apply threshold offset
    # --------------------------------------------------------

    final_threshold = int(
        round(
            smoothed_threshold
            + threshold_offset
        )
    )

    final_threshold = int(
        np.clip(
            final_threshold,
            1,
            255
        )
    )

    # --------------------------------------------------------
    # Create masks
    # --------------------------------------------------------

    raw_mask, clean_mask = (
        create_clean_mask(
            difference_gray,
            final_threshold,
            minimum_area
        )
    )

    # --------------------------------------------------------
    # Apply mask to original frame
    # --------------------------------------------------------

    rat_only = cv2.bitwise_and(
        current_frame,
        current_frame,
        mask=clean_mask
    )

    # --------------------------------------------------------
    # Colorize difference
    # --------------------------------------------------------

    difference_visualization = (
        cv2.applyColorMap(
            difference_gray,
            cv2.COLORMAP_TURBO
        )
    )

    # ========================================================
    # Build display panels
    # ========================================================

    original_panel = add_label(
        resize_to_panel(
            current_frame
        ),
        "Original"
    )

    difference_panel = add_label(
        resize_to_panel(
            difference_visualization
        ),
        f"Difference | Otsu: "
        f"{otsu_threshold:.0f}"
    )

    raw_mask_panel = add_label(
        resize_to_panel(
            raw_mask,
            is_mask=True
        ),
        f"Raw mask | Used: "
        f"{final_threshold}"
    )

    clean_mask_panel = add_label(
        resize_to_panel(
            clean_mask,
            is_mask=True
        ),
        f"Clean mask | Area: "
        f"{minimum_area}"
    )

    rat_panel = add_label(
        resize_to_panel(
            rat_only
        ),
        "Rat only"
    )

    background_panel = add_label(
        resize_to_panel(
            estimated_background
        ),
        "Estimated background"
    )

    # ========================================================
    # Create 3x2 split-screen
    # ========================================================

    top_row = cv2.hconcat([
        original_panel,
        difference_panel,
        raw_mask_panel
    ])

    bottom_row = cv2.hconcat([
        clean_mask_panel,
        rat_panel,
        background_panel
    ])

    split_screen = cv2.vconcat([
        top_row,
        bottom_row
    ])

    cv2.imshow(
        WINDOW_NAME,
        split_screen
    )

    # ========================================================
    # Keyboard controls
    # ========================================================

    key = cv2.waitKey(
        30 if paused else frame_delay
    ) & 0xFF

    # --------------------------------------------------------
    # Q / Esc: quit without exporting
    # --------------------------------------------------------

    if (
        key == ord("q")
        or key == 27
    ):

        break

    # --------------------------------------------------------
    # Space: pause / resume
    # --------------------------------------------------------

    if key == ord(" "):

        paused = not paused

    # --------------------------------------------------------
    # R: restart video
    # --------------------------------------------------------

    if key == ord("r"):

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        success, current_frame = (
            capture.read()
        )

        smoothed_threshold = None

        if not success:
            break

        continue

    # --------------------------------------------------------
    # E: export using CURRENT slider settings
    # --------------------------------------------------------

    if key == ord("e"):

        export_video(
            video_type=EXPORT_VIDEO_TYPE,
            minimum_area=minimum_area,
            threshold_offset=threshold_offset
        )

        # Keep viewer running after export.
        continue

    # --------------------------------------------------------
    # Move to next frame unless paused
    # --------------------------------------------------------

    if not paused:

        success, next_frame = (
            capture.read()
        )

        if not success:

            print(
                "End of video."
            )

            break

        current_frame = next_frame


# ============================================================
# Cleanup
# ============================================================

capture.release()
cv2.destroyAllWindows()

print()
print("Viewer closed.")

