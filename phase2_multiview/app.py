import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from model import SizeRegressor
from measurement import contour_dimensions
from preprocessing import preprocess_image

# Page configuration
st.set_page_config(
    page_title="Potato Size Estimator",
    page_icon="🥔",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained model weights"""
    if not os.path.exists("weights.npy"):
        st.error("❌ Model weights not found! Please train the model first by running: python train.py")
        st.stop()
    
    model = SizeRegressor()
    model.W = np.load("weights.npy")
    return model

model = load_model()

# Title and description
st.title("🥔 Potato Size Estimator")
st.markdown("""
Upload **3 views** of the same potato (top, left, right) to estimate its dimensions.
The app will predict the **length**, **width**, and **thickness** in millimeters.
""")

st.divider()

# Create three columns for image uploads
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📷 Top View")
    top_file = st.file_uploader("Upload top view", type=["jpg", "jpeg", "png"], key="top")
    if top_file:
        top_image = Image.open(top_file)
        st.image(top_image, caption="Top View", use_container_width=True)

with col2:
    st.subheader("📷 Left View")
    left_file = st.file_uploader("Upload left view", type=["jpg", "jpeg", "png"], key="left")
    if left_file:
        left_image = Image.open(left_file)
        st.image(left_image, caption="Left View", use_container_width=True)

with col3:
    st.subheader("📷 Right View")
    right_file = st.file_uploader("Upload right view", type=["jpg", "jpeg", "png"], key="right")
    if right_file:
        right_image = Image.open(right_file)
        st.image(right_image, caption="Right View", use_container_width=True)

st.divider()

# Process button
if st.button("🔍 Estimate Size", type="primary", use_container_width=True):
    # Check if all three images are uploaded
    if not all([top_file, left_file, right_file]):
        st.error("⚠️ Please upload all three views (top, left, right)")
    else:
        with st.spinner("Processing images..."):
            try:
                # Convert PIL images to numpy arrays
                top_array = np.array(top_image)
                left_array = np.array(left_image)
                right_array = np.array(right_image)
                
                # Save temporarily to process
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                
                top_path = os.path.join(temp_dir, "temp_top.jpg")
                left_path = os.path.join(temp_dir, "temp_left.jpg")
                right_path = os.path.join(temp_dir, "temp_right.jpg")
                
                cv2.imwrite(top_path, cv2.cvtColor(top_array, cv2.COLOR_RGB2BGR))
                cv2.imwrite(left_path, cv2.cvtColor(left_array, cv2.COLOR_RGB2BGR))
                cv2.imwrite(right_path, cv2.cvtColor(right_array, cv2.COLOR_RGB2BGR))
                
                # Extract contours
                top_contour = preprocess_image(top_path)
                left_contour = preprocess_image(left_path)
                right_contour = preprocess_image(right_path)
                
                # Check if contours were found
                if top_contour is None or left_contour is None or right_contour is None:
                    failed_views = []
                    if top_contour is None:
                        failed_views.append("top")
                    if left_contour is None:
                        failed_views.append("left")
                    if right_contour is None:
                        failed_views.append("right")
                    
                    st.error(f"❌ Could not detect potato in {', '.join(failed_views)} view(s). Please ensure:")
                    st.markdown("""
                    - The potato is clearly visible against a contrasting background
                    - The image is well-lit
                    - The potato fills most of the frame
                    """)
                else:
                    # Extract dimensions
                    top_dims = contour_dimensions(top_contour)
                    left_dims = contour_dimensions(left_contour)
                    right_dims = contour_dimensions(right_contour)
                    
                    # Build feature vector
                    length, width = top_dims
                    thickness = (left_dims[1] + right_dims[1]) / 2
                    features = np.array([[length, width, thickness]])
                    
                    # Predict
                    prediction = model.predict(features)[0]
                    
                    # Display results
                    st.success("✅ Estimation Complete!")
                    st.divider()
                    
                    # Display measured features
                    st.subheader("📐 Measured Features (from images)")
                    feat_col1, feat_col2, feat_col3 = st.columns(3)
                    with feat_col1:
                        st.metric("Top Length", f"{length:.1f} mm")
                    with feat_col2:
                        st.metric("Top Width", f"{width:.1f} mm")
                    with feat_col3:
                        st.metric("Avg Thickness", f"{thickness:.1f} mm")
                    
                    st.divider()
                    
                    # Display predictions
                    st.subheader("🎯 Predicted Actual Dimensions")
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    with pred_col1:
                        st.metric(
                            "Length",
                            f"{prediction[0]:.1f} mm",
                            delta=f"{prediction[0] - length:.1f} mm"
                        )
                    with pred_col2:
                        st.metric(
                            "Width",
                            f"{prediction[1]:.1f} mm",
                            delta=f"{prediction[1] - width:.1f} mm"
                        )
                    with pred_col3:
                        st.metric(
                            "Thickness",
                            f"{prediction[2]:.1f} mm",
                            delta=f"{prediction[2] - thickness:.1f} mm"
                        )
                    
                    # Summary
                    st.divider()
                    st.info(f"""
                    **Summary:** This potato is approximately **{prediction[0]:.1f} mm** long, 
                    **{prediction[1]:.1f} mm** wide, and **{prediction[2]:.1f} mm** thick.
                    """)
                    
                # Cleanup temp files
                for path in [top_path, left_path, right_path]:
                    if os.path.exists(path):
                        os.remove(path)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    ### How to use:
    1. **Take 3 photos** of the same potato:
       - One from the top
       - One from the left side
       - One from the right side
    
    2. **Upload** all three images
    
    3. Click **"Estimate Size"**
    
    ### Tips for best results:
    - Use a plain, contrasting background
    - Ensure good lighting
    - Keep the potato centered in the frame
    - Avoid shadows
    - Take photos from the same distance
    
    ### Model Information:
    - **Type:** Linear Regression
    - **Features:** Contour dimensions from 3 views
    - **Calibration:** MM_PER_PIXEL in measurement.py
    """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    from measurement import MM_PER_PIXEL
    st.info(f"**Current Calibration:**\n\nMM_PER_PIXEL = {MM_PER_PIXEL}")
    
    if st.button("🔄 Reload Model"):
        st.cache_resource.clear()
        st.rerun()