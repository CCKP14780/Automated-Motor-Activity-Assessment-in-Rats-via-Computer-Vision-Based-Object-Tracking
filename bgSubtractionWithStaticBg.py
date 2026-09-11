import cv2
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

VIDEO_NAME = "M8"

VIDEO_PATH = (
    rf"sleap-labeling\datasets\train\{VIDEO_NAME}.mov"
)

BACKGROUND_PATH = (
    rf"sleap-labeling\datasets\stillFrames\{VIDEO_NAME}_static_bg.jpg"
)


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
# Use "none" if you do not want to export anything.
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
# Image processing parameters
# ============================================================

BLUR_KERNEL_SIZE = 3       # Must be odd
CLOSE_KERNEL_SIZE = 5      # Must be odd
CLOSE_ITERATIONS = 1

# ============================================================
# Threshold configuration
# ============================================================

# Threshold method:
#   "otsu"      = global automatic threshold
#   "adaptive"  = local adaptive threshold
THRESHOLD_METHOD = "adaptive"

# Limits applied to Otsu's automatic threshold
MIN_AUTO_THRESHOLD = 3
MAX_AUTO_THRESHOLD = 50

# Adaptive threshold parameters.
# BLOCK_SIZE must be odd and greater than 1.
# Larger values use a larger local neighborhood.
ADAPTIVE_BLOCK_SIZE = 103

# Constant subtracted from the local mean.
# Increase this if too much background/noise becomes foreground.
ADAPTIVE_C = 3

# Threshold smoothing:
# Lower = more stable, higher = reacts faster.
# Used by Otsu mode. Adaptive mode calculates a local threshold
# independently for each frame.
THRESHOLD_SMOOTHING = 0.10

# Trackbar defaults
INITIAL_MIN_AREA = 100

# The offset slider ranges from -30 to +30.
INITIAL_THRESHOLD_OFFSET = -3
OFFSET_RANGE = 30


# ============================================================
# Display configuration
# ============================================================

PANEL_WIDTH = 360
PANEL_HEIGHT = 220

WINDOW_NAME = "Static Background Rat Segmentation"


# ============================================================
# Display functions
# ============================================================

def resize_to_panel(image, is_mask=False):
    """Resize an image while preserving its aspect ratio."""

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

    # Convert grayscale to BGR so all panels
    # have three channels.
    if resized.ndim == 2:
        resized = cv2.cvtColor(
            resized,
            cv2.COLOR_GRAY2BGR
        )

    panel = np.zeros(
        (PANEL_HEIGHT, PANEL_WIDTH, 3),
        dtype=np.uint8
    )

    x_offset = (
        PANEL_WIDTH - new_width
    ) // 2

    y_offset = (
        PANEL_HEIGHT - new_height
    ) // 2

    panel[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return panel


def add_label(image, text):
    """Add a label to the top of a display panel."""

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
        0.58,
        (0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return output


# ============================================================
# Segmentation functions
# ============================================================

def calculate_difference(
    frame,
    blurred_background
):
    """Calculate frame-to-background color difference."""

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

    # Use the strongest difference across
    # B, G, and R.
    difference_gray = np.max(
        color_difference,
        axis=2
    ).astype(np.uint8)

    return difference_gray


def calculate_otsu_threshold(
    difference_gray
):
    """Automatically calculate the foreground threshold."""

    otsu_threshold, _ = cv2.threshold(
        difference_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return float(
        np.clip(
            otsu_threshold,
            MIN_AUTO_THRESHOLD,
            MAX_AUTO_THRESHOLD
        )
    )


def calculate_adaptive_mask(
    difference_gray
):
    """
    Create a foreground mask using local adaptive thresholding.

    Each pixel is compared with the local mean intensity around it.
    This can handle uneven illumination or background differences
    better than one global threshold.
    """

    block_size = ADAPTIVE_BLOCK_SIZE

    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError(
            "ADAPTIVE_BLOCK_SIZE must be an odd integer greater than 1."
        )

    adaptive_mask = cv2.adaptiveThreshold(
        difference_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        ADAPTIVE_C
    )

    return adaptive_mask


def create_masks(
    difference_gray,
    threshold_value,
    minimum_area
):
    """Create raw and cleaned foreground masks."""

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

    # Find connected foreground regions.
    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            connected_mask,
            connectivity=8
        )
    )

    clean_mask = np.zeros_like(
        raw_mask
    )

    # Keep every component above
    # the minimum area.
    for label in range(
        1,
        component_count
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

    The current slider values are passed into this
    function, so the exported result matches the
    settings selected in the viewer.

    The output is always 1920x1080.
    """

    valid_types = {
        "original",
        "background",
        "difference",
        "raw_mask",
        "clean_mask",
        "rat_only"
    }

    if video_type == "none":

        print()
        print(
            "Export skipped "
            "(EXPORT_VIDEO_TYPE = 'none')."
        )

        return

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
        f"Method:     {THRESHOLD_METHOD}"
    )
    print(
        f"Threshold:  {threshold_offset:+d}"
    )
    print(
        f"Resolution: "
        f"{EXPORT_WIDTH} x {EXPORT_HEIGHT}"
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
    # Create video writer
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

        # Create masks using the selected threshold method.
        if THRESHOLD_METHOD == "adaptive":
            raw_mask = calculate_adaptive_mask(
                difference_gray
            )

            # The threshold offset is implemented as a small
            # morphological adjustment in adaptive mode.
            # Positive offset makes the mask stricter by removing
            # small foreground regions; negative offset is more permissive.
            if threshold_offset != 0:
                if threshold_offset > 0:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (3, 3)
                    )
                    raw_mask = cv2.erode(
                        raw_mask,
                        kernel,
                        iterations=min(threshold_offset, 5)
                    )
                else:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (3, 3)
                    )
                    raw_mask = cv2.dilate(
                        raw_mask,
                        kernel,
                        iterations=min(abs(threshold_offset), 5)
                    )

            used_threshold_text = "Adaptive"

        elif THRESHOLD_METHOD == "otsu":
            otsu_threshold = (
                calculate_otsu_threshold(
                    difference_gray
                )
            )

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

            raw_mask, _ = create_masks(
                difference_gray,
                final_threshold,
                minimum_area
            )

            used_threshold_text = str(final_threshold)

        else:
            raise ValueError(
                "THRESHOLD_METHOD must be 'otsu' or 'adaptive'."
            )

        # Clean the selected raw mask.
        _, clean_mask = create_masks(
            difference_gray,
            255,  # temporary threshold; raw_mask is supplied below
            minimum_area
        )

        # Re-apply morphology/component filtering directly to raw_mask.
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
        )

        connected_mask = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=CLOSE_ITERATIONS
        )

        component_count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                connected_mask,
                connectivity=8
            )
        )

        clean_mask = np.zeros_like(raw_mask)

        for label in range(1, component_count):
            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area >= minimum_area:
                clean_mask[labels == label] = 255

        # Create rat-only image
        rat_only = cv2.bitwise_and(
            frame,
            frame,
            mask=clean_mask
        )

        # Select requested output
        output_frame = (
            get_export_frame(
                video_type,
                frame,
                difference_gray,
                raw_mask,
                clean_mask,
                rat_only,
                background
            )
        )

        # ----------------------------------------------------
        # Resize to 1920x1080
        # ----------------------------------------------------

        output_frame = cv2.resize(
            output_frame,
            (
                EXPORT_WIDTH,
                EXPORT_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        # Write frame
        writer.write(
            output_frame
        )

        frame_count += 1

        # Print progress every 100 frames
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
# Load video and background
# ============================================================

capture = cv2.VideoCapture(
    VIDEO_PATH
)

background = cv2.imread(
    BACKGROUND_PATH
)

if not capture.isOpened():

    raise RuntimeError(
        f"Could not open video: "
        f"{VIDEO_PATH}"
    )

if background is None:

    capture.release()

    raise RuntimeError(
        f"Could not open background: "
        f"{BACKGROUND_PATH}"
    )


video_width = int(
    capture.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

video_height = int(
    capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
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
# Match background size to video size
# ============================================================

if background.shape[:2] != (
    video_height,
    video_width
):

    print(
        "Warning: resizing background image "
        "to match the video resolution."
    )

    background = cv2.resize(
        background,
        (
            video_width,
            video_height
        ),
        interpolation=cv2.INTER_AREA
    )


blurred_background = cv2.GaussianBlur(
    background,
    (
        BLUR_KERNEL_SIZE,
        BLUR_KERNEL_SIZE
    ),
    sigmaX=0
)


# ============================================================
# Create window and sliders
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


cv2.createTrackbar(
    "Min area",
    WINDOW_NAME,
    INITIAL_MIN_AREA,
    10000,
    lambda value: None
)


# Slider range: 0–60
# Actual threshold offset: -30 to +30

initial_offset_position = (
    INITIAL_THRESHOLD_OFFSET
    + OFFSET_RANGE
)


cv2.createTrackbar(
    "Threshold offset",
    WINDOW_NAME,
    initial_offset_position,
    OFFSET_RANGE * 2,
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
# Main processing loop
# ============================================================

while True:

    # --------------------------------------------------------
    # Get CURRENT slider values
    # --------------------------------------------------------

    minimum_area = cv2.getTrackbarPos(
        "Min area",
        WINDOW_NAME
    )

    threshold_offset = (
        cv2.getTrackbarPos(
            "Threshold offset",
            WINDOW_NAME
        )
        - OFFSET_RANGE
    )

    # --------------------------------------------------------
    # Compare current frame with static background
    # --------------------------------------------------------

    difference_gray = (
        calculate_difference(
            current_frame,
            blurred_background
        )
    )

    # --------------------------------------------------------
    # Create masks using selected threshold method
    # --------------------------------------------------------

    if THRESHOLD_METHOD == "adaptive":

        raw_mask = calculate_adaptive_mask(
            difference_gray
        )

        # Keep the existing slider useful in adaptive mode.
        if threshold_offset > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3)
            )
            raw_mask = cv2.erode(
                raw_mask,
                kernel,
                iterations=min(threshold_offset, 5)
            )

        elif threshold_offset < 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3)
            )
            raw_mask = cv2.dilate(
                raw_mask,
                kernel,
                iterations=min(abs(threshold_offset), 5)
            )

        # Apply the same closing + connected-component filtering
        # used by the normal mask pipeline.
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
        )

        connected_mask = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=CLOSE_ITERATIONS
        )

        component_count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                connected_mask,
                connectivity=8
            )
        )

        clean_mask = np.zeros_like(raw_mask)

        for label in range(1, component_count):

            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area >= minimum_area:
                clean_mask[
                    labels == label
                ] = 255

        used_threshold_text = "Adaptive"

    elif THRESHOLD_METHOD == "otsu":

        otsu_threshold = (
            calculate_otsu_threshold(
                difference_gray
            )
        )

        # ----------------------------------------------------
        # Smooth automatic threshold
        # ----------------------------------------------------

        if smoothed_threshold is None:
            smoothed_threshold = otsu_threshold

        elif not paused:
            smoothed_threshold = (
                (1.0 - THRESHOLD_SMOOTHING)
                * smoothed_threshold
                +
                THRESHOLD_SMOOTHING
                * otsu_threshold
            )

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

        raw_mask, clean_mask = (
            create_masks(
                difference_gray,
                final_threshold,
                minimum_area
            )
        )

        used_threshold_text = str(final_threshold)

    else:
        raise ValueError(
            "THRESHOLD_METHOD must be 'otsu' or 'adaptive'."
        )

    # --------------------------------------------------------
    # Create rat-only image
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
    # Create display panels
    # ========================================================

    original_panel = add_label(
        resize_to_panel(
            current_frame
        ),
        "Original video"
    )

    background_panel = add_label(
        resize_to_panel(
            background
        ),
        "Static background"
    )

    difference_panel = add_label(
        resize_to_panel(
            difference_visualization
        ),
        f"Difference | Method: "
        f"{THRESHOLD_METHOD}"
    )

    raw_mask_panel = add_label(
        resize_to_panel(
            raw_mask,
            is_mask=True
        ),
        f"Raw mask | {used_threshold_text}"
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

    # ========================================================
    # Create 3x2 split-screen
    # ========================================================

    top_row = cv2.hconcat([
        original_panel,
        background_panel,
        difference_panel
    ])

    bottom_row = cv2.hconcat([
        raw_mask_panel,
        clean_mask_panel,
        rat_panel
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
    # Q or Esc: quit WITHOUT exporting
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

        # Continue displaying the video
        # after export.
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

