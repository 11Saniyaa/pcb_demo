from pathlib import Path
import cv2
import csv


# -----------------------------
# 1. PATHS
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

image_path = (
    PROJECT_ROOT
    / "data"
    / "originals"
    / "pcb_06.jpg"
)

csv_path = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "annotations.csv"
)


# -----------------------------
# 2. LOAD ORIGINAL IMAGE
# -----------------------------

image = cv2.imread(str(image_path))

if image is None:
    print("Could not load image:", image_path)
    exit()


original_h, original_w = image.shape[:2]

print("Image loaded:", image_path.name)
print("Original size:", original_w, "x", original_h)


# -----------------------------
# 3. READ AND DRAW ANNOTATIONS
# -----------------------------

annotation_count = 0

with open(csv_path, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        # Only draw annotations belonging
        # to the current PCB image
        if row["image"] != image_path.name:
            continue


        component = row["component"]

        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])


        # Bottom-right corner
        x2 = x + w
        y2 = y + h


        # Draw box on ORIGINAL image
        cv2.rectangle(
            image,
            (x, y),
            (x2, y2),
            (0, 255, 0),
            3
        )


        # Draw label
        cv2.putText(
            image,
            component,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        annotation_count += 1


print("Annotations found:", annotation_count)


# -----------------------------
# 4. RESIZE ONLY FOR DISPLAY
# -----------------------------

max_display_width = 1200
max_display_height = 750


scale_w = max_display_width / original_w
scale_h = max_display_height / original_h


# Never enlarge smaller images
display_scale = min(
    scale_w,
    scale_h,
    1.0
)


display_w = int(
    original_w * display_scale
)

display_h = int(
    original_h * display_scale
)


display_image = cv2.resize(
    image,
    (display_w, display_h),
    interpolation=cv2.INTER_AREA
)


print(
    "Display size:",
    display_w,
    "x",
    display_h
)

print(
    "Display scale:",
    round(display_scale, 4)
)


# -----------------------------
# 5. DISPLAY
# -----------------------------

cv2.imshow(
    "PCB Annotations",
    display_image
)

print("\nClick the image window.")
print("Press ANY KEY to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()

print("\nDone.")