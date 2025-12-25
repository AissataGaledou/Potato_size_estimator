import os
import csv
from preprocessing import preprocess_image
from measurement import contour_dimensions

dataset_dir = "dataset"
images_dir = os.path.join(dataset_dir, "images")
labels_path = os.path.join(dataset_dir, "labels.csv")

print("=" * 60)
print("🔍 DATASET STRUCTURE DEBUG")
print("=" * 60)

# 1. Check if paths exist
print(f"\n1️⃣ Checking paths:")
print(f"   Dataset dir exists: {os.path.exists(dataset_dir)}")
print(f"   Images dir exists: {os.path.exists(images_dir)}")
print(f"   Labels file exists: {os.path.exists(labels_path)}")

# 2. List all image files
valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
print(f"\n2️⃣ Image files found:")
if os.path.exists(images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(valid_extensions)]
    print(f"   Total image files: {len(image_files)}")
    for i, f in enumerate(image_files[:5]):  # Show first 5
        print(f"   - {f}")
    if len(image_files) > 5:
        print(f"   ... and {len(image_files) - 5} more")
else:
    print("   ❌ Images directory not found!")

# 3. Parse labels
print(f"\n3️⃣ Labels in CSV:")
labels = {}
if os.path.exists(labels_path):
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"   CSV headers: {headers}")
        for row in reader:
            labels[row["id"]] = [
                float(row["length_mm"]),
                float(row["width_mm"]),
                float(row["thickness_mm"])
            ]
    print(f"   Total labels: {len(labels)}")
    for i, (pid, dims) in enumerate(list(labels.items())[:3]):
        print(f"   - {pid}: {dims}")
else:
    print("   ❌ Labels file not found!")

# 4. Test filename parsing
print(f"\n4️⃣ Testing filename parsing:")
if os.path.exists(images_dir):
    for file in image_files[:3]:
        print(f"   File: {file}")
        try:
            # Current parsing logic - remove extension first
            name_without_ext = os.path.splitext(file)[0]
            pid, view = name_without_ext.rsplit("_", 1)
            print(f"      → Parsed ID: '{pid}', View: '{view}'")
            print(f"      → ID in labels? {pid in labels}")
        except Exception as e:
            print(f"      ❌ Parse error: {e}")

# 5. Test contour extraction
print(f"\n5️⃣ Testing contour extraction:")
if os.path.exists(images_dir) and image_files:
    test_file = image_files[0]
    test_path = os.path.join(images_dir, test_file)
    print(f"   Testing: {test_file}")
    
    contour = preprocess_image(test_path)
    if contour is not None:
        width_mm, height_mm = contour_dimensions(contour)
        print(f"   ✅ Contour extracted!")
        print(f"      Dimensions: {width_mm:.2f} x {height_mm:.2f} mm")
    else:
        print(f"   ❌ No contour found (image may be blank/corrupted)")

# 6. Simulate feature extraction
print(f"\n6️⃣ Simulating full feature extraction:")
features = {}
valid_count = 0
invalid_count = 0

if os.path.exists(images_dir):
    for file in image_files:
        try:
            name_without_ext = os.path.splitext(file)[0]
            pid, view = name_without_ext.rsplit("_", 1)
            view = view.lower()  # Normalize view name
            contour = preprocess_image(os.path.join(images_dir, file))
            
            if contour is None:
                invalid_count += 1
                continue

            features.setdefault(pid, {})[view] = contour_dimensions(contour)
            valid_count += 1
        except Exception as e:
            print(f"   ❌ Error processing {file}: {e}")
            invalid_count += 1

print(f"   Valid contours: {valid_count}")
print(f"   Invalid/failed: {invalid_count}")
print(f"   Unique potatoes: {len(features)}")

# 7. Check complete samples
print(f"\n7️⃣ Complete samples (all 3 views + label):")
complete_samples = 0
for pid, views in features.items():
    has_all_views = all(v in views for v in ("top", "left", "right"))
    has_label = pid in labels
    
    if has_all_views and has_label:
        complete_samples += 1
        if complete_samples <= 3:
            print(f"   ✅ {pid}: top={views['top']}, left={views['left']}, right={views['right']}")
    elif has_all_views:
        print(f"   ⚠️  {pid}: has all views but NO LABEL")
    elif has_label:
        missing = [v for v in ("top", "left", "right") if v not in views]
        print(f"   ⚠️  {pid}: has label but missing views: {missing}")

print(f"\n   Total complete samples: {complete_samples}")

print("\n" + "=" * 60)
print("🎯 DIAGNOSIS SUMMARY")
print("=" * 60)
if complete_samples == 0:
    print("❌ No complete samples found!")
    print("\nPossible causes:")
    print("1. ID mismatch between filenames and labels.csv")
    print("2. Missing views (need top, left, right for each potato)")
    print("3. Contour extraction failing (blank/corrupted images)")
    print("4. Filename format doesn't match expected pattern")
else:
    print(f"✅ Found {complete_samples} complete samples!")
    print("   Your pipeline should work now.")