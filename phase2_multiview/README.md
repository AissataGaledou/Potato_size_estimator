#  Potato Size Estimator - Phase 2: Multi-view Size Estimation

A computer vision system that estimates potato dimensions (length, width, thickness) from three-view images using OpenCV and linear regression.

##  Overview

This project estimates the physical dimensions of potatoes using:
- **Computer Vision**: OpenCV for contour detection and measurement
- **Multi-view Analysis**: Combining top, left, and right views
- **Linear Regression**: NumPy-based model for size prediction
- **Web Interface**: Streamlit app for easy image upload and estimation

##  Features

- ✅ Multi-view image processing (top, left, right)
- ✅ Automatic contour detection and segmentation
- ✅ Camera calibration support
- ✅ Linear regression model for size estimation
- ✅ Interactive web interface with Streamlit
- ✅ Real-time size prediction

##  Project Structure

```
root_crop_size_estimator/phase2_multiview/
├── model.py              # Linear regression model
├── preprocessing.py      # Image processing and dataset loading
├── measurement.py        # Camera calibration and contour measurement
├── train.py             # Model training script
├── evaluate.py          # Model evaluation script
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

##  Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/root_crop_size_estimator.git
cd root_crop_size_estimator/phase2_multiview
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

##  Dataset Preparation

Create your dataset with the following structure:

```
dataset/
├── images/
│   ├── potato_001_top.jpg
│   ├── potato_001_left.jpg
│   ├── potato_001_right.jpg
│   ├── potato_002_top.jpg
│   └── ...
└── labels.csv
```

**labels.csv format:**
```csv
id,length_mm,width_mm,thickness_mm
potato_001,110.0,60.0,60.0
potato_002,70.0,50.0,60.0
potato_003,40.0,90.0,70.0
```

##  Camera Calibration

Before training, calibrate your camera by adjusting `MM_PER_PIXEL` in `measurement.py`:

1. Take a photo of an object with known dimensions (e.g., ruler)
2. Measure it in pixels
3. Calculate: `MM_PER_PIXEL = actual_size_mm / measured_pixels`

Example:
```python
# In measurement.py
MM_PER_PIXEL = 0.1  # Adjust based on your camera setup
```

##  Training

Train the model with your dataset:

```bash
python train.py
```

This will:
- Load and process all images
- Extract features from contours
- Train a linear regression model
- Save weights to `weights.npy`

##  Evaluation

Evaluate model performance:

```bash
python evaluate.py
```

Output shows mean errors for each dimension:
```
Mean Length Error: 2.5 mm
Mean Width Error: 2.1 mm
Mean Thickness Error: 1.8 mm
```

##  Web Interface

Launch the Streamlit app:

```bash
streamlit run app.py
```

Then:
1. Upload 3 views of a potato (top, left, right)
2. Click "Estimate Size"
3. View predicted dimensions

## 🔧 Troubleshooting

### No samples detected
- Check that image filenames match the pattern: `potato_XXX_view.jpg`
- Verify images are in `dataset/images/`
- Ensure IDs in `labels.csv` match image filenames

### High prediction errors
- Recalibrate `MM_PER_PIXEL` in `measurement.py`
- Ensure consistent lighting and background in images
- Check that all 3 views are from the same distance

### Contour detection fails
- Use plain, contrasting background
- Improve lighting conditions
- Center potato in frame

## 🛠️ Debug Tools

Run diagnostic scripts to identify issues:

```bash
# Check dataset structure and image loading
python debug_dataset.py

# Analyze prediction quality
python debug_predictions.py

# Verify files in dataset
python check_files.py
```

##  Technical Details

### Architecture
- **Image Processing**: Otsu's thresholding + contour detection
- **Feature Extraction**: Bounding box dimensions from 3 views
- **Model**: Linear regression (closed-form solution using pseudo-inverse)
- **Input**: [top_length, top_width, avg_thickness] (3 features)
- **Output**: [actual_length, actual_width, actual_thickness] (3 labels)

### Why Linear Regression?
- Simple and interpretable
- Fast training and inference
- No deep learning dependencies (PyTorch unavailable)
- Sufficient for dimension mapping task

##  Future Improvements

- [ ] Add more training data (50-100+ samples)
- [ ] Implement train/test split
- [ ] Add data augmentation
- [ ] Support single-view estimation
- [ ] Add batch processing
- [ ] Export results to CSV
- [ ] Add confidence scores


##  Author

Aissata Galedou

##  Acknowledgments

- OpenCV for computer vision tools
- Streamlit for the web interface
- NumPy for numerical computing