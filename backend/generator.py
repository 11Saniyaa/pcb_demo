from pathlib import Path
import csv

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINALS_DIR = PROJECT_ROOT / "data" / "originals"
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "annotations" / "annotations.csv"
DEMO_OUTPUT_DIR = PROJECT_ROOT / "data" / "augmented" / "demo"

SUPPORTED_COMPONENTS = {"ic", "resistor", "capacitor", "diode"}

COMPONENT_CONFIG = {
    name: {
        "threshold_mode": "auto",
        "kernel_size": 3,
        "open_iterations": 1,
        "close_iterations": 2,
        "dilate_iterations": 1,
    }
    for name in SUPPORTED_COMPONENTS
}

DEFECTS = {
    "rotate_05": {
        "id": "rotate_05",
        "label": "Slight rotation (+5°)",
        "industry": "Skewed placement",
        "type": "rotate",
        "parameter": "+5 degrees",
        "suffix": "rotate_05",
        "angle": 5,
    },
    "rotate_15": {
        "id": "rotate_15",
        "label": "Severe rotation (+15°)",
        "industry": "Gross rotational misplacement",
        "type": "rotate",
        "parameter": "+15 degrees",
        "suffix": "rotate_15",
        "angle": 15,
    },
    "flip_horizontal": {
        "id": "flip_horizontal",
        "label": "Horizontal flip",
        "industry": "Wrong polarity / reversed part",
        "type": "flip",
        "parameter": "horizontal",
        "suffix": "flip_horizontal",
    },
    "move_10px": {
        "id": "move_10px",
        "label": "Shift +10 px",
        "industry": "Fine misalignment",
        "type": "move",
        "parameter": "+10 px x-direction",
        "suffix": "move_10px",
        "shift_x": 10,
        "shift_y": 0,
    },
    "move_25px": {
        "id": "move_25px",
        "label": "Shift +25 px",
        "industry": "Coarse misalignment",
        "type": "move",
        "parameter": "+25 px x-direction",
        "suffix": "move_25px",
        "shift_x": 25,
        "shift_y": 0,
    },
}


def _row_value(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    raise KeyError(keys)


def load_all_annotations():
    if not ANNOTATIONS_PATH.exists():
        return []

    rows = []
    with ANNOTATIONS_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if not any((raw or {}).values()):
                continue
            try:
                image_name = _row_value(raw, "image", "image", "file")
                component = _row_value(raw, "component", "component", "class").lower()
                x = int(_row_value(raw, "x"))
                y = int(_row_value(raw, "y"))
                width = int(_row_value(raw, "width"))
                height = int(_row_value(raw, "height"))
            except (KeyError, ValueError):
                continue

            if component not in SUPPORTED_COMPONENTS:
                continue

            rows.append(
                {
                    "image": image_name,
                    "board": Path(image_name).stem,
                    "component": component,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )
    return rows


_BOARD_CACHE = None
_BOARD_CACHE_MTIME = None


def list_boards():
    global _BOARD_CACHE, _BOARD_CACHE_MTIME
    mtime = ANNOTATIONS_PATH.stat().st_mtime if ANNOTATIONS_PATH.exists() else None
    if _BOARD_CACHE is not None and _BOARD_CACHE_MTIME == mtime:
        return _BOARD_CACHE

    annotations = load_all_annotations()
    by_board = {}
    for row in annotations:
        by_board.setdefault(row["board"], []).append(row)

    boards = []
    for image_path in sorted(ORIGINALS_DIR.glob("pcb_*.png")):
        board_id = image_path.stem
        items = by_board.get(board_id, [])
        counts = {}
        components = []
        for item in items:
            counts[item["component"]] = counts.get(item["component"], 0) + 1
            components.append(
                {
                    "component": item["component"],
                    "instance": counts[item["component"]],
                    "x": item["x"],
                    "y": item["y"],
                    "width": item["width"],
                    "height": item["height"],
                }
            )

        image = cv2.imread(str(image_path))
        height, width = (image.shape[:2] if image is not None else (0, 0))
        boards.append(
            {
                "id": board_id,
                "file": image_path.name,
                "width": int(width),
                "height": int(height),
                "annotated": bool(components),
                "components": components,
            }
        )
    _BOARD_CACHE = boards
    _BOARD_CACHE_MTIME = mtime
    return boards


def get_board(board_id):
    for board in list_boards():
        if board["id"] == board_id:
            return board
    return None


def original_image_path(board_id):
    path = ORIGINALS_DIR / f"{board_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Board image not found: {board_id}")
    return path


def clean_binary_mask(binary_mask, config):
    kernel = np.ones((config["kernel_size"], config["kernel_size"]), dtype=np.uint8)
    cleaned = cv2.morphologyEx(
        binary_mask, cv2.MORPH_OPEN, kernel, iterations=config["open_iterations"]
    )
    return cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, kernel, iterations=config["close_iterations"]
    )


def contour_score(contour, mask_shape):
    roi_height, roi_width = mask_shape
    roi_area = roi_height * roi_width
    area = cv2.contourArea(contour)
    if area <= 0:
        return -1

    area_ratio = area / roi_area
    if area_ratio < 0.02 or area_ratio > 0.95:
        return -1

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return -1

    contour_x = moments["m10"] / moments["m00"]
    contour_y = moments["m01"] / moments["m00"]
    center_x = roi_width / 2
    center_y = roi_height / 2
    distance = np.sqrt((contour_x - center_x) ** 2 + (contour_y - center_y) ** 2)
    maximum_distance = np.sqrt(center_x**2 + center_y**2)
    if maximum_distance == 0:
        return -1
    return (area_ratio * 0.7) + ((1 - distance / maximum_distance) * 0.3)


def create_candidate_mask(gray, threshold_type, config):
    otsu_value, binary_mask = cv2.threshold(
        gray, 0, 255, threshold_type + cv2.THRESH_OTSU
    )
    binary_mask = clean_binary_mask(binary_mask, config)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, -1, otsu_value

    best_contour = None
    best_score = -1
    for contour in contours:
        score = contour_score(contour, binary_mask.shape)
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None, -1, otsu_value

    final_mask = np.zeros_like(binary_mask)
    cv2.drawContours(final_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    if config["dilate_iterations"] > 0:
        final_mask = cv2.dilate(
            final_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=config["dilate_iterations"],
        )
    return final_mask, best_score, otsu_value


def create_grabcut_mask(roi, config):
    roi_height, roi_width = roi.shape[:2]
    if roi_height < 5 or roi_width < 5:
        return None

    grabcut_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    margin_x = max(2, int(roi_width * 0.02))
    margin_y = max(2, int(roi_height * 0.02))
    rectangle_width = roi_width - (2 * margin_x)
    rectangle_height = roi_height - (2 * margin_y)
    if rectangle_width <= 0 or rectangle_height <= 0:
        return None

    try:
        cv2.grabCut(
            roi,
            grabcut_mask,
            (margin_x, margin_y, rectangle_width, rectangle_height),
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None

    final_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    kernel = np.ones((config["kernel_size"], config["kernel_size"]), dtype=np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if config["dilate_iterations"] > 0:
        final_mask = cv2.dilate(
            final_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=config["dilate_iterations"],
        )

    ratio = cv2.countNonZero(final_mask) / (roi_width * roi_height)
    if ratio < 0.02 or ratio > 0.98:
        return None
    return final_mask


def segment_component(roi, component_name):
    config = COMPONENT_CONFIG[component_name]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    candidates = []

    normal_mask, normal_score, normal_threshold = create_candidate_mask(
        gray, cv2.THRESH_BINARY, config
    )
    inverse_mask, inverse_score, inverse_threshold = create_candidate_mask(
        gray, cv2.THRESH_BINARY_INV, config
    )
    if normal_mask is not None:
        candidates.append((normal_mask, normal_threshold, "normal", normal_score))
    if inverse_mask is not None:
        candidates.append((inverse_mask, inverse_threshold, "inverse", inverse_score))
    if candidates:
        return max(candidates, key=lambda item: item[3])

    grabcut_mask = create_grabcut_mask(roi, config)
    if grabcut_mask is not None:
        return grabcut_mask, -1, "grabcut", 0.5
    return None, -1, "failed", -1


def remove_original_component(original_image, component_mask, x, y, width, height):
    full_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    full_mask[y : y + height, x : x + width] = component_mask
    full_mask = cv2.dilate(full_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return cv2.inpaint(original_image, full_mask, 3, cv2.INPAINT_TELEA)


def paste_component(base_image, component_roi, component_mask, new_x, new_y):
    result = base_image.copy()
    component_height, component_width = component_roi.shape[:2]
    if (
        new_x < 0
        or new_y < 0
        or new_x + component_width > result.shape[1]
        or new_y + component_height > result.shape[0]
    ):
        return None

    destination = result[
        new_y : new_y + component_height, new_x : new_x + component_width
    ]
    inverse_mask = cv2.bitwise_not(component_mask)
    destination_background = cv2.bitwise_and(destination, destination, mask=inverse_mask)
    component_foreground = cv2.bitwise_and(
        component_roi, component_roi, mask=component_mask
    )
    result[new_y : new_y + component_height, new_x : new_x + component_width] = cv2.add(
        destination_background, component_foreground
    )
    return result


def rotate_component_expanded(component_roi, component_mask, angle):
    height, width = component_roi.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(rotation_matrix[0, 0])
    sine = abs(rotation_matrix[0, 1])
    new_width = int(np.ceil((height * sine) + (width * cosine)))
    new_height = int(np.ceil((height * cosine) + (width * sine)))
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]

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
    _, rotated_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)
    return rotated_roi, rotated_mask


def _build_augmentation(roi, mask, x, y, width, height, defect):
    if defect["type"] == "rotate":
        rotated_roi, rotated_mask = rotate_component_expanded(roi, mask, defect["angle"])
        rotated_height, rotated_width = rotated_roi.shape[:2]
        return {
            "image": rotated_roi,
            "mask": rotated_mask,
            "new_x": x - (rotated_width - width) // 2,
            "new_y": y - (rotated_height - height) // 2,
        }
    if defect["type"] == "flip":
        return {
            "image": cv2.flip(roi, 1),
            "mask": cv2.flip(mask, 1),
            "new_x": x,
            "new_y": y,
        }
    if defect["type"] == "move":
        return {
            "image": roi.copy(),
            "mask": mask.copy(),
            "new_x": x + int(defect.get("shift_x", 0)),
            "new_y": y + int(defect.get("shift_y", 0)),
        }
    raise ValueError(f"Unsupported defect type: {defect['type']}")


def encode_jpeg(image, max_width=1400, quality=85):
    output = image
    height, width = image.shape[:2]
    if max_width and width > max_width:
        scale = max_width / width
        output = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode image")
    return buffer.tobytes()


def generate_defect(board_id, component, instance, defect_id):
    component = component.strip().lower()
    if component not in SUPPORTED_COMPONENTS:
        raise ValueError(f"Unsupported component: {component}")
    if defect_id not in DEFECTS:
        raise ValueError(f"Unknown defect: {defect_id}")
    if instance < 1:
        raise ValueError("Instance must be >= 1")

    board = get_board(board_id)
    if board is None:
        raise FileNotFoundError(f"Unknown board: {board_id}")

    matches = [item for item in board["components"] if item["component"] == component]
    if not matches:
        raise ValueError(f"No {component} annotation on {board_id}")
    if instance > len(matches):
        raise ValueError(
            f"{board_id} has only {len(matches)} {component} annotation(s)"
        )

    annotation = matches[instance - 1]
    image = cv2.imread(str(original_image_path(board_id)))
    if image is None:
        raise FileNotFoundError(f"Could not read {board_id}.png")

    image_height, image_width = image.shape[:2]
    x = annotation["x"]
    y = annotation["y"]
    width = annotation["width"]
    height = annotation["height"]
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid bounding box")
    if x + width > image_width or y + height > image_height:
        raise ValueError("Bounding box is outside the image")

    roi = image[y : y + height, x : x + width].copy()
    mask, _threshold, method, score = segment_component(roi, component)
    if mask is None:
        raise RuntimeError("Could not segment that component")

    foreground_ratio = cv2.countNonZero(mask) / (width * height)
    if foreground_ratio < 0.02 or foreground_ratio > 0.98:
        raise RuntimeError("Component mask looks unreliable")

    background = remove_original_component(image, mask, x, y, width, height)
    defect = DEFECTS[defect_id]
    augmentation = _build_augmentation(roi, mask, x, y, width, height, defect)
    result = paste_component(
        background,
        augmentation["image"],
        augmentation["mask"],
        int(augmentation["new_x"]),
        int(augmentation["new_y"]),
    )
    if result is None:
        raise RuntimeError(
            "Defect crossed the image boundary. Try another fault or part."
        )

    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{board_id}_{component}_{instance:02d}_{defect['suffix']}.png"
    output_path = DEMO_OUTPUT_DIR / filename
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError("Could not save generated image")

    new_height, new_width = augmentation["image"].shape[:2]
    return {
        "board": board_id,
        "component": component,
        "instance": instance,
        "defect": defect,
        "filename": filename,
        "segmentation_method": method,
        "mask_score": round(float(score), 3),
        "original_box": {"x": x, "y": y, "width": width, "height": height},
        "defect_box": {
            "x": int(augmentation["new_x"]),
            "y": int(augmentation["new_y"]),
            "width": int(new_width),
            "height": int(new_height),
        },
        "image_size": {"width": image_width, "height": image_height},
        "output_path": output_path,
    }
