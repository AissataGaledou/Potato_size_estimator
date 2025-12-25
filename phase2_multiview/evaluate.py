import numpy as np
from preprocessing import preprocess_dataset
from model import SizeRegressor

X, y = preprocess_dataset("dataset")

model = SizeRegressor()
model.W = np.load("weights.npy")

pred = model.predict(X)
error = np.abs(pred - y)

print("\n📊 Evaluation Results")
print(f"Mean Length Error: {error[:,0].mean():.2f} mm")
print(f"Mean Width Error: {error[:,1].mean():.2f} mm")
print(f"Mean Thickness Error: {error[:,2].mean():.2f} mm")
