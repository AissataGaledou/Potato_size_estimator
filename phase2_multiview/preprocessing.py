import os
import cv2
import csv
import numpy as np
from measurement import contour_dimensions

# Note: If features are too large/small compared to labels,
# adjust MM_PER_PIXEL in measurement.py

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)

def preprocess_dataset(dataset_dir):
    images_dir = os.path.join(dataset_dir, "images")
    labels_path = os.path.join(dataset_dir, "labels.csv")

    # Load labels
    labels = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"].strip()
            labels[pid] = [
                float(row["length_mm"]),
                float(row["width_mm"]),
                float(row["thickness_mm"])
            ]

    # Extract features from images
    # Accept multiple image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    features = {}
    skipped = []
    
    for file in os.listdir(images_dir):
        if not file.endswith(valid_extensions):
            continue

        try:
            # Remove extension and parse
            # Support: potato_001_top.jpg, potato_001_top.png, etc.
            name_without_ext = os.path.splitext(file)[0]
            pid, view = name_without_ext.rsplit("_", 1)
            pid = pid.strip()
            view = view.strip().lower()  # Normalize view name
            
            # Extract contour
            contour = preprocess_image(os.path.join(images_dir, file))
            if contour is None:
                skipped.append(f"{file} (no contour)")
                continue

            # Store dimensions
            features.setdefault(pid, {})[view] = contour_dimensions(contour)
            
        except ValueError as e:
            skipped.append(f"{file} (parse error: {e})")
            continue

    # Build training dataset
    X, y = [], []
    missing_labels = []
    incomplete_views = []
    
    for pid, views in features.items():
        # Check if we have label
        if pid not in labels:
            missing_labels.append(pid)
            continue
        
        # Check if we have all 3 views
        if not all(v in views for v in ("top", "left", "right")):
            missing = [v for v in ("top", "left", "right") if v not in views]
            incomplete_views.append(f"{pid} (missing: {', '.join(missing)})")
            continue
        
        # Extract features
        length, width = views["top"]
        thickness = (views["left"][1] + views["right"][1]) / 2
        X.append([length, width, thickness])
        y.append(labels[pid])

    # Print diagnostics
    print(f"\n📊 Dataset Processing Summary:")
    print(f"   Labels loaded: {len(labels)}")
    print(f"   Images found: {len([f for f in os.listdir(images_dir) if f.endswith(valid_extensions)])}")
    print(f"   Unique potatoes with features: {len(features)}")
    print(f"   Valid samples: {len(X)}")
    
    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} images:")
        for s in skipped[:5]:
            print(f"   - {s}")
        if len(skipped) > 5:
            print(f"   ... and {len(skipped) - 5} more")
    
    if missing_labels:
        print(f"\n⚠️  {len(missing_labels)} potatoes have images but no labels:")
        for ml in missing_labels[:5]:
            print(f"   - {ml}")
        if len(missing_labels) > 5:
            print(f"   ... and {len(missing_labels) - 5} more")
    
    if incomplete_views:
        print(f"\n⚠️  {len(incomplete_views)} potatoes have incomplete views:")
        for iv in incomplete_views[:5]:
            print(f"   - {iv}")
        if len(incomplete_views) > 5:
            print(f"   ... and {len(incomplete_views) - 5} more")

    return np.array(X), np.array(y)