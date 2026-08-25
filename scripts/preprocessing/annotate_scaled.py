from pathlib import Path
import csv

import cv2


# ==================================================
# 1. PROJECT PATHS
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINALS_DIR = PROJECT_ROOT / "data" / "originals"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
CSV_PATH = ANNOTATION_DIR / "annotations.csv"

ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. IMAGES AND COMPONENTS
# ==================================================
IMAGE_NAMES = [
    "pcb_09.jpg",
    "pcb_10.jpg",
    "pcb_11.jpg",
]

COMPONENTS = [
    "IC",
    "resistor",
    "capacitor",
    "diode",
]

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 750

all_annotations = []


# ==================================================
# 3. PROCESS EACH IMAGE
# ==================================================
for image_name in IMAGE_NAMES:

    image_path = ORIGINALS_DIR / image_name
    image = cv2.imread(str(image_path))

    if image is None:
        print("\nCould not load image:", image_path)
        continue

    original_h, original_w = image.shape[:2]

    print("\n======================================")
    print("Annotating:", image_name)
    print("Original size:", original_w, "x", original_h)
    print("======================================")


    # ----------------------------------------------
    # Calculate display scale
    # ----------------------------------------------
    scale_w = MAX_DISPLAY_WIDTH / original_w
    scale_h = MAX_DISPLAY_HEIGHT / original_h

    # Do not enlarge smaller images
    scale = min(scale_w, scale_h, 1.0)

    display_w = round(original_w * scale)
    display_h = round(original_h * scale)

    if scale < 1.0:
        display_image = cv2.resize(
            image,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA
        )
    else:
        display_image = image.copy()

    print("Display size:", display_w, "x", display_h)
    print("Display scale:", round(scale, 4))


    # ==================================================
    # 4. SELECT COMPONENTS
    # ==================================================
    for component in COMPONENTS:

        print(f"\nSelect one {component} in {image_name}")
        print("Press ENTER or SPACE after selecting.")
        print("Press C to skip if the component is unavailable.")

        window_name = f"{image_name} - Select {component}"

        x_display, y_display, w_display, h_display = cv2.selectROI(
            window_name,
            display_image,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow(window_name)

        if w_display == 0 or h_display == 0:
            print(f"Skipped {component}")
            continue


        # ----------------------------------------------
        # Convert back to original-image coordinates
        # ----------------------------------------------
        x = round(x_display / scale)
        y = round(y_display / scale)
        width = round(w_display / scale)
        height = round(h_display / scale)


        # Keep bounding box inside the image
        x = max(0, min(x, original_w - 1))
        y = max(0, min(y, original_h - 1))

        width = min(width, original_w - x)
        height = min(height, original_h - y)


        all_annotations.append([
            image_name,
            component,
            x,
            y,
            width,
            height,
        ])

        print(
            f"Saved {component}: "
            f"x={x}, y={y}, width={width}, height={height}"
        )

    cv2.destroyAllWindows()


# ==================================================
# 5. SAVE ANNOTATIONS
# ==================================================
file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0

with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "image",
            "component",
            "x",
            "y",
            "width",
            "height",
        ])

    writer.writerows(all_annotations)


print("\n======================================")
print("Annotation completed")
print("Rows saved:", len(all_annotations))
print("CSV location:", CSV_PATH)
print("======================================")