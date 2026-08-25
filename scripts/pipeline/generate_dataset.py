from pathlib import Path
import subprocess
import sys
import csv

import cv2
import numpy as np

# ==================================================
# SETTINGS
# ==================================================

# PCBs that will be processed automatically
PCB_NAMES = [
    
    "pcb_01",
    "pcb_04"
    
]


# When the script is started normally,
# run it separately for every PCB.
if len(sys.argv) == 1:

    print("\n======================================")
    print("      PCB BATCH PROCESSING STARTED")
    print("======================================")

    for pcb_name in PCB_NAMES:

        print("\nProcessing:", pcb_name)

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                pcb_name,
            ],
            check=True,
        )

    print("\n======================================")
    print("      PCB BATCH PROCESSING FINISHED")
    print("======================================\n")

    sys.exit(0)


# During each individual run, obtain the PCB
# name passed by the batch section above.
PCB_NAME = sys.argv[1]


SUPPORTED_COMPONENTS = {
    "ic",
    "resistor",
    "capacitor",
    "diode",
}

COMPONENT_CONFIG = {
    "ic": {
        "threshold_mode": "auto",
        "kernel_size": 3,
        "open_iterations": 1,
        "close_iterations": 2,
        "dilate_iterations": 1,
    },
    "resistor": {
        "threshold_mode": "auto",
        "kernel_size": 3,
        "open_iterations": 1,
        "close_iterations": 2,
        "dilate_iterations": 1,
    },
    "capacitor": {
        "threshold_mode": "auto",
        "kernel_size": 3,
        "open_iterations": 1,
        "close_iterations": 2,
        "dilate_iterations": 1,
    },
    "diode": {
    "threshold_mode": "auto",
    "kernel_size": 3,
    "open_iterations": 1,
    "close_iterations": 1,
    "dilate_iterations": 0,
},
}

SHOW_PREVIEWS = True
SAVE_DEBUG_IMAGES = True

# Red box settings
BOX_COLOUR = (0, 0, 255)
BOX_THICKNESS = 3
BOX_PADDING_RATIO = 0.15
MINIMUM_BOX_PADDING = 12


print("\n======================================")
print("  PCB COMPONENT PIPELINE STARTED")
print("======================================\n")


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINALS_DIR = (
    PROJECT_ROOT
    / "data"
    / "originals"
)

ANNOTATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "annotations.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "augmented"
    / PCB_NAME
)

DEBUG_FOLDER = OUTPUT_FOLDER / "debug"

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "augmented"
    / "augmentation_metadata.csv"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

if SAVE_DEBUG_IMAGES:
    DEBUG_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )


def find_image_path():
    """
    Find the selected PCB regardless of whether
    it is stored as JPG, JPEG or PNG.
    """

    possible_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
    ]

    for extension in possible_extensions:
        candidate = (
            ORIGINALS_DIR
            / f"{PCB_NAME}{extension}"
        )

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {PCB_NAME} with "
        f"extensions: {possible_extensions}"
    )


image_path = find_image_path()

print("PCB           :", PCB_NAME)
print("Image         :", image_path)
print("Annotations   :", ANNOTATIONS_PATH)
print("Output folder :", OUTPUT_FOLDER)
print("Metadata      :", METADATA_PATH)


# ==================================================
# 2. LOAD ORIGINAL IMAGE
# ==================================================

print("\n[1/7] Loading original PCB image...")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"OpenCV could not load: {image_path}"
    )

image_height, image_width = image.shape[:2]

print("Image loaded successfully")
print("Dimensions:", image_width, "x", image_height)


# ==================================================
# 3. LOAD ANNOTATIONS
# ==================================================

def load_annotations(
    csv_file,
    selected_image,
):
    matched_annotations = []

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {csv_file}"
        )

    with open(
        csv_file,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "image",
            "component",
            "x",
            "y",
            "width",
            "height",
        }

        existing_columns = set(
            reader.fieldnames or []
        )

        if not required_columns.issubset(
            existing_columns
        ):
            raise ValueError(
                "annotations.csv does not contain "
                "all required columns"
            )

        for row in reader:
            row_image = (
                row["image"]
                .strip()
            )

            component = (
                row["component"]
                .strip()
                .lower()
            )

            if (
                row_image == selected_image
                and component
                in SUPPORTED_COMPONENTS
            ):
                matched_annotations.append({
                    "image": row_image,
                    "component": component,
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                })

    return matched_annotations


print("\n[2/7] Reading annotations...")

annotations = load_annotations(
    ANNOTATIONS_PATH,
    image_path.name,
)

if not annotations:
    raise ValueError(
        "No supported component annotations "
        f"found for {image_path.name}"
    )

print(
    f"Found {len(annotations)} annotation(s)"
)

for annotation in annotations:
    print(
        f"{annotation['component']:10s} "
        f"({annotation['x']}, "
        f"{annotation['y']}, "
        f"{annotation['width']}, "
        f"{annotation['height']})"
    )

found_components = {
    annotation["component"]
    for annotation in annotations
}

missing_components = (
    SUPPORTED_COMPONENTS
    - found_components
)

print(
    "Components found:",
    sorted(found_components)
)

if missing_components:
    print(
        "WARNING: Missing annotations:",
        sorted(missing_components)
    )


# ==================================================
# 4. SEGMENTATION FUNCTIONS
# ==================================================

def clean_binary_mask(
    binary_mask,
    config,
):
    kernel_size = config["kernel_size"]

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    cleaned = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=config[
            "open_iterations"
        ],
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config[
            "close_iterations"
        ],
    )

    return cleaned


def contour_score(
    contour,
    mask_shape,
):
    roi_height, roi_width = mask_shape

    roi_area = roi_height * roi_width
    contour_area = cv2.contourArea(contour)

    if contour_area <= 0:
        return -1

    area_ratio = contour_area / roi_area

    if (
        area_ratio < 0.02
        or area_ratio > 0.95
    ):
        return -1

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return -1

    contour_x = (
        moments["m10"]
        / moments["m00"]
    )

    contour_y = (
        moments["m01"]
        / moments["m00"]
    )

    roi_center_x = roi_width / 2
    roi_center_y = roi_height / 2

    distance = np.sqrt(
        (contour_x - roi_center_x) ** 2
        + (contour_y - roi_center_y) ** 2
    )

    maximum_distance = np.sqrt(
        roi_center_x ** 2
        + roi_center_y ** 2
    )

    if maximum_distance == 0:
        return -1

    centre_score = (
        1 - distance / maximum_distance
    )

    return (
        area_ratio * 0.7
        + centre_score * 0.3
    )


def create_candidate_mask(
    gray,
    threshold_type,
    config,
):
    otsu_value, binary_mask = cv2.threshold(
        gray,
        0,
        255,
        threshold_type + cv2.THRESH_OTSU,
    )

    binary_mask = clean_binary_mask(
        binary_mask,
        config,
    )

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None, -1, otsu_value

    best_contour = None
    best_score = -1

    for contour in contours:
        score = contour_score(
            contour,
            binary_mask.shape,
        )

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None, -1, otsu_value

    final_mask = np.zeros_like(
        binary_mask
    )

    cv2.drawContours(
        final_mask,
        [best_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    if config["dilate_iterations"] > 0:
        final_mask = cv2.dilate(
            final_mask,
            np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
            iterations=config[
                "dilate_iterations"
            ],
        )

    return (
        final_mask,
        best_score,
        otsu_value,
    )


def create_grabcut_mask(
    roi,
    config,
):
    roi_height, roi_width = roi.shape[:2]

    if roi_height < 5 or roi_width < 5:
        return None

    grabcut_mask = np.zeros(
        (roi_height, roi_width),
        dtype=np.uint8,
    )

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    margin_x = max(
        2,
        int(roi_width * 0.02),
    )

    margin_y = max(
        2,
        int(roi_height * 0.02),
    )

    rectangle_width = (
        roi_width - 2 * margin_x
    )

    rectangle_height = (
        roi_height - 2 * margin_y
    )

    if (
        rectangle_width <= 0
        or rectangle_height <= 0
    ):
        return None

    rectangle = (
        margin_x,
        margin_y,
        rectangle_width,
        rectangle_height,
    )

    try:
        cv2.grabCut(
            roi,
            grabcut_mask,
            rectangle,
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )

    except cv2.error:
        return None

    final_mask = np.where(
        (grabcut_mask == cv2.GC_FGD)
        | (
            grabcut_mask
            == cv2.GC_PR_FGD
        ),
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones(
        (
            config["kernel_size"],
            config["kernel_size"],
        ),
        dtype=np.uint8,
    )

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    if config["dilate_iterations"] > 0:
        final_mask = cv2.dilate(
            final_mask,
            np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
            iterations=config[
                "dilate_iterations"
            ],
        )

    foreground_ratio = (
        cv2.countNonZero(final_mask)
        / (roi_width * roi_height)
    )

    if (
        foreground_ratio < 0.02
        or foreground_ratio > 0.98
    ):
        return None

    return final_mask


def segment_component(
    roi,
    component_name,
):
    config = COMPONENT_CONFIG[
        component_name
    ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )

    threshold_mode = config[
        "threshold_mode"
    ]

    candidates = []

    if threshold_mode in {
        "auto",
        "normal",
    }:
        (
            normal_mask,
            normal_score,
            normal_threshold,
        ) = create_candidate_mask(
            gray,
            cv2.THRESH_BINARY,
            config,
        )

        if normal_mask is not None:
            candidates.append({
                "mask": normal_mask,
                "score": normal_score,
                "threshold": normal_threshold,
                "method": "normal",
            })

    if threshold_mode in {
        "auto",
        "inverse",
    }:
        (
            inverse_mask,
            inverse_score,
            inverse_threshold,
        ) = create_candidate_mask(
            gray,
            cv2.THRESH_BINARY_INV,
            config,
        )

        if inverse_mask is not None:
            candidates.append({
                "mask": inverse_mask,
                "score": inverse_score,
                "threshold": inverse_threshold,
                "method": "inverse",
            })

    if candidates:
        best_candidate = max(
            candidates,
            key=lambda item: item["score"],
        )

        return (
            best_candidate["mask"],
            best_candidate["threshold"],
            best_candidate["method"],
            best_candidate["score"],
        )

    print(
        f"Otsu failed for {component_name}. "
        "Trying GrabCut..."
    )

    grabcut_mask = create_grabcut_mask(
        roi,
        config,
    )

    if grabcut_mask is not None:
        return (
            grabcut_mask,
            -1,
            "grabcut",
            0.5,
        )

    return None, -1, "failed", -1


# ==================================================
# 5. IMAGE MANIPULATION
# ==================================================

def remove_original_component(
    original_image,
    component_mask,
    x,
    y,
    width,
    height,
):
    full_mask = np.zeros(
        original_image.shape[:2],
        dtype=np.uint8,
    )

    full_mask[
        y:y + height,
        x:x + width,
    ] = component_mask

    full_mask = cv2.dilate(
        full_mask,
        np.ones(
            (5, 5),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    background = cv2.inpaint(
        original_image,
        full_mask,
        3,
        cv2.INPAINT_TELEA,
    )

    return background


def paste_component(
    base_image,
    component_roi,
    component_mask,
    new_x,
    new_y,
):
    result = base_image.copy()

    component_height, component_width = (
        component_roi.shape[:2]
    )

    if (
        new_x < 0
        or new_y < 0
        or (
            new_x + component_width
            > result.shape[1]
        )
        or (
            new_y + component_height
            > result.shape[0]
        )
    ):
        return None

    destination = result[
        new_y:new_y + component_height,
        new_x:new_x + component_width,
    ]

    inverse_mask = cv2.bitwise_not(
        component_mask
    )

    destination_background = cv2.bitwise_and(
        destination,
        destination,
        mask=inverse_mask,
    )

    component_foreground = cv2.bitwise_and(
        component_roi,
        component_roi,
        mask=component_mask,
    )

    combined = cv2.add(
        destination_background,
        component_foreground,
    )

    result[
        new_y:new_y + component_height,
        new_x:new_x + component_width,
    ] = combined

    return result


def rotate_component_expanded(
    component_roi,
    component_mask,
    angle,
):
    height, width = component_roi.shape[:2]

    center = (
        width / 2,
        height / 2,
    )

    rotation_matrix = (
        cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )
    )

    cosine = abs(rotation_matrix[0, 0])
    sine = abs(rotation_matrix[0, 1])

    new_width = int(
        np.ceil(
            height * sine
            + width * cosine
        )
    )

    new_height = int(
        np.ceil(
            height * cosine
            + width * sine
        )
    )

    rotation_matrix[0, 2] += (
        new_width / 2 - center[0]
    )

    rotation_matrix[1, 2] += (
        new_height / 2 - center[1]
    )

    rotated_roi = cv2.warpAffine(
        component_roi,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    rotated_mask = cv2.warpAffine(
        component_mask,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    _, rotated_mask = cv2.threshold(
        rotated_mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return rotated_roi, rotated_mask


def draw_defect_box(
    generated_image,
    component_name,
    augmentation_type,
    new_x,
    new_y,
    width,
    height,
):
    """
    Draw a red box larger than the transformed
    component directly on the demo output.
    """

    boxed_image = generated_image.copy()

    current_height, current_width = (
        boxed_image.shape[:2]
    )

    padding = max(
        MINIMUM_BOX_PADDING,
        round(
            max(width, height)
            * BOX_PADDING_RATIO
        ),
    )

    box_x1 = max(
        0,
        new_x - padding,
    )

    box_y1 = max(
        0,
        new_y - padding,
    )

    box_x2 = min(
        current_width - 1,
        new_x + width + padding,
    )

    box_y2 = min(
        current_height - 1,
        new_y + height + padding,
    )

    cv2.rectangle(
        boxed_image,
        (box_x1, box_y1),
        (box_x2, box_y2),
        BOX_COLOUR,
        BOX_THICKNESS,
    )

    label = (
        f"DEFECT: "
        f"{component_name.upper()} "
        f"{augmentation_type.upper()}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        text_thickness,
    )

    label_x1 = box_x1
    label_y2 = box_y1

    label_y1 = max(
        0,
        label_y2 - text_height - 12,
    )

    label_x2 = min(
        current_width - 1,
        label_x1 + text_width + 10,
    )

    # Filled red background behind text
    cv2.rectangle(
        boxed_image,
        (label_x1, label_y1),
        (label_x2, label_y2),
        BOX_COLOUR,
        thickness=cv2.FILLED,
    )

    text_y = max(
        text_height,
        label_y2 - 6,
    )

    cv2.putText(
        boxed_image,
        label,
        (label_x1 + 5, text_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    return boxed_image


# ==================================================
# 6. METADATA
# ==================================================

METADATA_HEADER = [
    "original_image",
    "augmented_image",
    "component",
    "augmentation_type",
    "parameter",
    "old_x",
    "old_y",
    "new_x",
    "new_y",
    "width",
    "height",
]

new_metadata_rows = []


def update_metadata_file():
    existing_rows = []

    if METADATA_PATH.exists():
        with open(
            METADATA_PATH,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                if (
                    row.get("original_image")
                    != image_path.name
                ):
                    existing_rows.append([
                        row.get(column, "")
                        for column
                        in METADATA_HEADER
                    ])

    with open(
        METADATA_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(METADATA_HEADER)
        writer.writerows(existing_rows)
        writer.writerows(new_metadata_rows)


# ==================================================
# 7. PROCESS COMPONENTS
# ==================================================

print("\n[3/7] Processing components...")

component_counts = {}
generated_results = []

for annotation in annotations:
    component_name = annotation[
        "component"
    ]

    component_counts[component_name] = (
        component_counts.get(
            component_name,
            0,
        ) + 1
    )

    instance_number = component_counts[
        component_name
    ]

    x = annotation["x"]
    y = annotation["y"]
    width = annotation["width"]
    height = annotation["height"]

    print("\n--------------------------------------")
    print(
        f"Processing {component_name} "
        f"instance {instance_number}"
    )
    print("--------------------------------------")

    # ----------------------------------------------
    # Validate bounding box
    # ----------------------------------------------
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        print("Invalid bounding box. Skipping.")
        continue

    # ----------------------------------------------
    # Crop component ROI
    # ----------------------------------------------
    roi = image[
        y:y + height,
        x:x + width,
    ].copy()

    print("ROI cropped:", roi.shape)

    # ----------------------------------------------
    # Segment component
    # ----------------------------------------------
    (
        mask,
        threshold_value,
        segmentation_method,
        mask_score,
    ) = segment_component(
        roi,
        component_name,
    )

    if mask is None:
        print(
            "Could not create mask. Skipping."
        )
        continue

    foreground_ratio = (
        cv2.countNonZero(mask)
        / (width * height)
    )

    print("Mask generated")
    print(
        "Method:",
        segmentation_method,
    )
    print(
        "Score:",
        round(mask_score, 3),
    )
    print(
        "Foreground ratio:",
        round(foreground_ratio, 3),
    )

    if threshold_value >= 0:
        print(
            "Otsu threshold:",
            round(threshold_value, 2),
        )

    # ----------------------------------------------
    # GrabCut fallback
    # ----------------------------------------------
    # Only run GrabCut when Otsu covers too
    # little or too much of the ROI.
    if (
        foreground_ratio < 0.02
        or foreground_ratio > 0.98
    ):
        print(
            "Suspicious Otsu mask. "
            "Trying GrabCut fallback..."
        )

        fallback_mask = create_grabcut_mask(
            roi,
            COMPONENT_CONFIG[component_name],
        )

        if fallback_mask is None:
            print(
                f"GrabCut also failed for "
                f"{component_name}. Skipping."
            )
            continue

        mask = fallback_mask

        foreground_ratio = (
            cv2.countNonZero(mask)
            / (width * height)
        )

        segmentation_method = (
            "grabcut-fallback"
        )

        mask_score = 0.5

        print(
            "GrabCut fallback successful"
        )
        print(
            "New foreground ratio:",
            round(foreground_ratio, 3),
        )

    else:
        print(
            "Otsu mask accepted:",
            round(foreground_ratio, 3),
        )

    # ----------------------------------------------
    # Save segmentation debug images
    # ----------------------------------------------
    # This must be outside the if/else above,
    # but still inside the annotation loop.
    if SAVE_DEBUG_IMAGES:
        debug_prefix = (
            f"{PCB_NAME}_"
            f"{component_name}_"
            f"{instance_number:02d}"
        )

        extracted_preview = (
            cv2.bitwise_and(
                roi,
                roi,
                mask=mask,
            )
        )

        cv2.imwrite(
            str(
                DEBUG_FOLDER
                / f"{debug_prefix}_roi.png"
            ),
            roi,
        )

        cv2.imwrite(
            str(
                DEBUG_FOLDER
                / f"{debug_prefix}_mask.png"
            ),
            mask,
        )

        cv2.imwrite(
            str(
                DEBUG_FOLDER
                / (
                    f"{debug_prefix}_"
                    f"extracted.png"
                )
            ),
            extracted_preview,
        )

    # ----------------------------------------------
    # Remove original component
    # ----------------------------------------------
    background = remove_original_component(
        image,
        mask,
        x,
        y,
        width,
        height,
    )

    # Keep your existing augmentation code here.
    # It must remain indented inside this loop.
    augmentations = []

    # Your rotate, flip and move augmentation
    # definitions continue below this line.
    # --------------------------------------------------
    # 1. ROTATE 5 DEGREES
    # --------------------------------------------------
    rotated_5_roi, rotated_5_mask = (
        rotate_component_expanded(
            roi,
            mask,
            5,
        )
    )

    rotated_5_height, rotated_5_width = (
        rotated_5_roi.shape[:2]
    )

    augmentations.append({
        "image": rotated_5_roi,
        "mask": rotated_5_mask,
        "type": "rotate",
        "parameter": "+5 degrees",
        "suffix": "rotate_05",
        "new_x": (
            x
            - (rotated_5_width - width) // 2
        ),
        "new_y": (
            y
            - (rotated_5_height - height) // 2
        ),
    })

    # --------------------------------------------------
    # 2. HORIZONTAL FLIP
    # --------------------------------------------------
    augmentations.append({
        "image": cv2.flip(roi, 1),
        "mask": cv2.flip(mask, 1),
        "type": "flip",
        "parameter": "horizontal",
        "suffix": "flip_horizontal",
        "new_x": x,
        "new_y": y,
    })

    # --------------------------------------------------
    # 3. MOVE 10 PIXELS
    # --------------------------------------------------
    augmentations.append({
        "image": roi.copy(),
        "mask": mask.copy(),
        "type": "move",
        "parameter": "+10 px x-direction",
        "suffix": "move_10px",
        "new_x": x + 10,
        "new_y": y,
    })

    # --------------------------------------------------
    # 4. MOVE 25 PIXELS
    # --------------------------------------------------
    augmentations.append({
        "image": roi.copy(),
        "mask": mask.copy(),
        "type": "move",
        "parameter": "+25 px x-direction",
        "suffix": "move_25px",
        "new_x": x + 25,
        "new_y": y,
    })

    # --------------------------------------------------
    # 5. ROTATE 15 DEGREES
    # --------------------------------------------------
    rotated_15_roi, rotated_15_mask = (
        rotate_component_expanded(
            roi,
            mask,
            15,
        )
    )

    rotated_15_height, rotated_15_width = (
        rotated_15_roi.shape[:2]
    )

    augmentations.append({
        "image": rotated_15_roi,
        "mask": rotated_15_mask,
        "type": "rotate",
        "parameter": "+15 degrees",
        "suffix": "rotate_15",
        "new_x": (
            x
            - (rotated_15_width - width) // 2
        ),
        "new_y": (
            y
            - (rotated_15_height - height) // 2
        ),
    })

    generated_for_component = 0

    # ==================================================
    # PASTE, DRAW RED BOX AND SAVE
    # ==================================================
    for augmentation in augmentations:
        augmented_component = (
            augmentation["image"]
        )

        augmented_mask = (
            augmentation["mask"]
        )

        new_x = int(
            augmentation["new_x"]
        )

        new_y = int(
            augmentation["new_y"]
        )

        result = paste_component(
            background,
            augmented_component,
            augmented_mask,
            new_x,
            new_y,
        )

        if result is None:
            print(
                f"Skipped {augmentation['suffix']}: "
                "component crossed image boundary"
            )
            continue

        (
            augmented_height,
            augmented_width,
        ) = augmented_component.shape[:2]

        # Draw the larger red defect box directly
        # on the generated demo image.
        boxed_result = draw_defect_box(
            generated_image=result,
            component_name=component_name,
            augmentation_type=(
                augmentation["type"]
            ),
            new_x=new_x,
            new_y=new_y,
            width=augmented_width,
            height=augmented_height,
        )

        filename = (
            f"{PCB_NAME}_"
            f"{component_name}_"
            f"{instance_number:02d}_"
            f"{augmentation['suffix']}.png"
        )

        output_path = (
            OUTPUT_FOLDER / filename
        )

        # Save the generated defective image
        # with the red demo box.
        saved = cv2.imwrite(
            str(output_path),
            boxed_result,
        )

        if not saved:
            print(
                "Could not save:",
                output_path,
            )
            continue

        print("Generated:", filename)

        new_metadata_rows.append([
            image_path.name,
            filename,
            component_name,
            augmentation["type"],
            augmentation["parameter"],
            x,
            y,
            new_x,
            new_y,
            augmented_width,
            augmented_height,
        ])

        generated_results.append({
            "name": filename,
            "component": component_name,
            "augmentation": (
                augmentation["type"]
            ),
            "result": boxed_result,
        })

        generated_for_component += 1

    print(
        f"Generated {generated_for_component}/5 "
        f"images for {component_name}"
    )


# ==================================================
# 8. SAVE METADATA
# ==================================================

print("\n[4/7] Saving metadata...")

update_metadata_file()

print("Metadata saved:", METADATA_PATH)


# ==================================================
# 9. SUMMARY
# ==================================================

print("\n[5/7] Generation summary")

if not generated_results:
    print("No augmented images were generated.")

else:
    for item in generated_results:
        print(
            f"{item['component']:10s} "
            f"-> {item['name']}"
        )


# ==================================================
# 10. PREVIEW
# ==================================================

if SHOW_PREVIEWS and generated_results:
    print("\n[6/7] Opening previews...")
    print("Press any key for the next image.")

    max_preview_width = 1000
    max_preview_height = 700

    for index, item in enumerate(
        generated_results,
        start=1,
    ):
        generated_image = item["result"]

        result_height, result_width = (
            generated_image.shape[:2]
        )

        preview_scale = min(
            max_preview_width / result_width,
            max_preview_height / result_height,
            1.0,
        )

        if preview_scale < 1.0:
            preview = cv2.resize(
                generated_image,
                None,
                fx=preview_scale,
                fy=preview_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            preview = generated_image.copy()

        window_name = (
            f"{index}/"
            f"{len(generated_results)} - "
            f"{item['name']}"
        )

        cv2.imshow(
            window_name,
            preview,
        )

        cv2.waitKey(0)

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    cv2.destroyAllWindows()

else:
    print(
        "\n[6/7] Preview disabled "
        "or no results."
    )


# ==================================================
# 11. FINISHED
# ==================================================

print("\n[7/7] Finished")

print("\n======================================")
print("  PCB COMPONENT PIPELINE COMPLETED")
print("======================================")

print("Selected PCB      :", PCB_NAME)
print("Original unchanged:", image_path)
print("Annotations found :", len(annotations))
print(
    "Images expected   :",
    len(annotations) * 5,
)
print(
    "Images generated  :",
    len(generated_results),
)
print("Output folder     :", OUTPUT_FOLDER)
print("Metadata file     :", METADATA_PATH)

print("\nPROGRAM FINISHED\n")