import streamlit as st
import cv2
import numpy as np
from estimate_size import detect_coin, detect_potato_contour, compute_weight, annotate_image, COIN_DIAMETER_MM

st.set_page_config(page_title="Potato Size Estimator", layout="wide")

st.title("Potato Size & Weight Estimator")
st.markdown("### Using 1 Ghana Cedi Coin as Reference Scale")
st.write("Upload an image containing a potato and a 1 cedi coin placed together.")

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Detection Parameters")
    
    st.subheader("Coin Detection")
    dp = st.slider("dp (resolution)", 1.0, 2.0, 1.2, 0.1)
    param1 = st.slider("param1 (edge threshold)", 50, 200, 100)
    param2 = st.slider("param2 (circle threshold)", 10, 100, 30)
    
    st.markdown("---")
    st.info("💡 **Tips:**\n"
            "- Use flat, top-down photo\n"
            "- Place coin near potato\n"
            "- Good lighting helps\n"
            "- Avoid shadows")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader(" Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Read image
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                caption="📷 Original Image", use_container_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with st.spinner("🔍 Detecting coin and potato..."):
        # ---- Detect coin ----
        coin = detect_coin(gray)
        if coin is None:
            st.error("❌ No coin detected! Please ensure the coin is clearly visible.")
            st.stop()

        cx, cy, r = coin
        coin_px = r * 2
        scale = COIN_DIAMETER_MM / coin_px

        # ---- Detect potato ----
        potato = detect_potato_contour(img, coin_position=coin)
        if potato is None:
            st.error("❌ Potato not detected! Try better lighting or clearer image.")
            st.stop()

        if len(potato) < 5:
            st.error("❌ Not enough contour points to fit ellipse.")
            st.stop()

        # Fit ellipse with corrected major/minor ordering
        ellipse = cv2.fitEllipse(potato)
        (_, _), (axis1, axis2), _ = ellipse
        
        # CRITICAL FIX: Ensure major >= minor
        major_px = max(axis1, axis2)
        minor_px = min(axis1, axis2)

        major_mm = major_px * scale
        minor_mm = minor_px * scale
        
        # More realistic thickness estimate
        thickness_mm = 0.6 * minor_mm

        volume_cm3, weight_g = compute_weight(major_mm, minor_mm, thickness_mm)

    # Create annotated image
    measurements = {
        "Scale": f"{scale:.4f} mm/px",
        "Coin": f"{coin_px} px",
        "Major": f"{major_mm:.1f} mm",
        "Minor": f"{minor_mm:.1f} mm",
        "Thickness": f"{thickness_mm:.1f} mm",
        "Volume": f"{volume_cm3:.1f} cm³",
        "Weight": f"{weight_g:.1f} g"
    }
    
    annotated = annotate_image(img, coin=coin, contour=potato, measurements=measurements)

    with col2:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 
                caption="📊 Annotated Result", use_container_width=True)

    # Display results
    st.success("✅ Estimation Completed!")
    
    st.markdown("### 📏 Measurements")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Major Axis", f"{major_mm:.1f} mm", 
                 help="Longest dimension of the potato")
        st.metric("Minor Axis", f"{minor_mm:.1f} mm",
                 help="Shortest dimension of the potato")
    
    with metric_col2:
        st.metric("Est. Thickness", f"{thickness_mm:.1f} mm",
                 help="Estimated depth (60% of minor axis)")
        st.metric("Est. Volume", f"{volume_cm3:.1f} cm³",
                 help="Ellipsoid volume approximation")
    
    with metric_col3:
        st.metric("Est. Weight", f"{weight_g:.1f} g",
                 help="Using potato density ≈ 1.05 g/cm³")
        st.metric("Scale Factor", f"{scale:.4f} mm/px",
                 help="Calibration from coin")

    # Technical details
    with st.expander("🔬 Technical Details"):
        st.write(f"**Coin Detection:**")
        st.write(f"- Position: ({cx}, {cy})")
        st.write(f"- Radius: {r} px")
        st.write(f"- Diameter: {coin_px} px")
        st.write(f"\n**Potato Analysis:**")
        st.write(f"- Contour points: {len(potato)}")
        st.write(f"- Contour area: {cv2.contourArea(potato):.0f} px²")
        st.write(f"- Major axis (px): {major_px:.1f}")
        st.write(f"- Minor axis (px): {minor_px:.1f}")
        st.write(f"\n**Weight Calculation:**")
        st.write(f"- Assumed density: 1.05 g/cm³")
        st.write(f"- Weight: {weight_g/1000:.3f} kg")
    
    # Download button
    _, center_col, _ = st.columns([1, 1, 1])
    with center_col:
        # Convert to bytes for download
        _, buffer = cv2.imencode('.jpg', annotated)
        st.download_button(
            label=" Download Annotated Image",
            data=buffer.tobytes(),
            file_name="potato_measurement.jpg",
            mime="image/jpeg"
        )

else:
    st.info(" Please upload an image to begin estimation.")

# Footer
st.markdown("---")
st.caption("⚠️ **Note:** Measurements are approximate. For best results, use flat top-down photos with good lighting. "
           "Weight estimation assumes average potato density of 1.05 g/cm³.")