from preprocessing import preprocess_dataset
from model import SizeRegressor
import numpy as np

X, y = preprocess_dataset("dataset")

print("✅ Dataset processed")
print("Samples:", len(X))

model = SizeRegressor()
model.fit(X, y)

np.save("weights.npy", model.W)
print("✅ Model trained")
