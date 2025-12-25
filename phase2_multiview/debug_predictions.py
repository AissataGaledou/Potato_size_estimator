import numpy as np
from preprocessing import preprocess_dataset
from model import SizeRegressor

X, y = preprocess_dataset("dataset")

model = SizeRegressor()
model.W = np.load("weights.npy")

pred = model.predict(X)

print("=" * 70)
print("🔍 PREDICTION ANALYSIS")
print("=" * 70)

print("\n1️⃣ Input Features (X) - Extracted from Images:")
print(f"   Shape: {X.shape}")
print(f"   Min values: {X.min(axis=0)}")
print(f"   Max values: {X.max(axis=0)}")
print(f"   Mean values: {X.mean(axis=0)}")
print("\n   Sample features:")
for i in range(min(3, len(X))):
    print(f"   Sample {i}: length={X[i,0]:.1f}mm, width={X[i,1]:.1f}mm, thickness={X[i,2]:.1f}mm")

print("\n2️⃣ Ground Truth Labels (y):")
print(f"   Shape: {y.shape}")
print(f"   Min values: {y.min(axis=0)}")
print(f"   Max values: {y.max(axis=0)}")
print(f"   Mean values: {y.mean(axis=0)}")
print("\n   Sample labels:")
for i in range(min(3, len(y))):
    print(f"   Sample {i}: length={y[i,0]:.1f}mm, width={y[i,1]:.1f}mm, thickness={y[i,2]:.1f}mm")

print("\n3️⃣ Model Predictions:")
print(f"   Shape: {pred.shape}")
print(f"   Min values: {pred.min(axis=0)}")
print(f"   Max values: {pred.max(axis=0)}")
print(f"   Mean values: {pred.mean(axis=0)}")
print("\n   Sample predictions:")
for i in range(min(3, len(pred))):
    print(f"   Sample {i}: length={pred[i,0]:.1f}mm, width={pred[i,1]:.1f}mm, thickness={pred[i,2]:.1f}mm")

print("\n4️⃣ Detailed Comparison (Input → Truth vs Prediction):")
print("=" * 70)
for i in range(len(X)):
    print(f"\nSample {i}:")
    print(f"  Input Features:    [{X[i,0]:.1f}, {X[i,1]:.1f}, {X[i,2]:.1f}] mm")
    print(f"  True Label:        [{y[i,0]:.1f}, {y[i,1]:.1f}, {y[i,2]:.1f}] mm")
    print(f"  Predicted:         [{pred[i,0]:.1f}, {pred[i,1]:.1f}, {pred[i,2]:.1f}] mm")
    print(f"  Error:             [{abs(pred[i,0]-y[i,0]):.1f}, {abs(pred[i,1]-y[i,1]):.1f}, {abs(pred[i,2]-y[i,2]):.1f}] mm")

print("\n" + "=" * 70)
print("🎯 DIAGNOSIS")
print("=" * 70)

# Check if features and labels are in same scale
if X.mean() > 100 and y.mean() < 100:
    print("⚠️  SCALE MISMATCH DETECTED!")
    print(f"   Features are large (mean={X.mean():.1f}mm)")
    print(f"   Labels are small (mean={y.mean():.1f}mm)")
    print("\n   Possible causes:")
    print("   1. MM_PER_PIXEL calibration is wrong")
    print("   2. Features are in pixels, labels in mm (or vice versa)")
    print("   3. Labels in CSV are incorrect")
elif X.mean() < 10 and y.mean() > 50:
    print("⚠️  INVERTED SCALE DETECTED!")
    print(f"   Features are tiny (mean={X.mean():.1f}mm)")
    print(f"   Labels are normal (mean={y.mean():.1f}mm)")
    print("\n   Possible cause: MM_PER_PIXEL is too small")
else:
    print("✅ Features and labels are in similar scale")
    print(f"   Feature mean: {X.mean():.1f}mm")
    print(f"   Label mean: {y.mean():.1f}mm")