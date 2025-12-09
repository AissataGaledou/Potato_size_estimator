import cv2
import numpy as np

COIN_DIAMETER_MM = 26.0   # 1 Ghana Cedi coin


def detect_coin(gray):
    """Detect the 1 Ghana Cedi coin using Hough Circle Transform."""
    gray_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=120
    )

    if circles is None:
        return None
    
    circles = np.round(circles[0, :]).astype("int")
    # Sort by radius (largest first) to prefer actual coin
    circles = sorted(circles.tolist(), key=lambda x: x[2], reverse=True)
    x, y, r = circles[0]
    return (x, y, r)


def detect_potato_contour(img, coin_position=None):
    """
    Detect potato contour using adaptive thresholding + GrabCut.
    
    Args:
        img: BGR color image
        coin_position: (x, y, r) of coin to exclude from detection
        
    Returns:
        Largest valid potato contour or None
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Adaptive thresholding
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Mask out coin area if provided
    if coin_position is not None:
        coin_x, coin_y, coin_r = coin_position
        cv2.circle(thresh, (coin_x, coin_y), int(coin_r * 1.3), 0, -1)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) == 0:
        return None
    
    # Filter valid contours
    valid_contours = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        
        # Minimum area
        if area < 15000:
            continue
        
        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            continue
        
        # Check solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.75:
            continue
        
        # Check distance from coin
        if coin_position is not None:
            coin_x, coin_y, coin_r = coin_position
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - coin_x)**2 + (cy - coin_y)**2)
                if dist < coin_r * 1.5:
                    continue
        
        valid_contours.append(cnt)
    
    if len(valid_contours) == 0:
        return None
    
    # Get largest valid contour
    potato = max(valid_contours, key=cv2.contourArea)
    
    # Optional: Refine with GrabCut
    try:
        mask_grabcut = np.zeros(gray.shape, np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        x, y, w, h = cv2.boundingRect(potato)
        padding = 20
        rect = (max(0, x - padding), max(0, y - padding), 
                min(img.shape[1] - x + padding, w + 2*padding), 
                min(img.shape[0] - y + padding, h + 2*padding))
        
        cv2.grabCut(img, mask_grabcut, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask_refined = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 1).astype('uint8')
        
        if coin_position is not None:
            cv2.circle(mask_refined, (coin_x, coin_y), int(coin_r * 1.3), 0, -1)
        
        refined_cnts, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if refined_cnts:
            potato_refined = max(refined_cnts, key=cv2.contourArea)
            if cv2.contourArea(potato_refined) > 10000:
                potato = potato_refined
    except:
        pass
    
    return potato


def compute_weight(major_mm, minor_mm, thickness_mm):
    """
    Compute volume and weight of potato approximated as ellipsoid.
    
    Args:
        major_mm: Major axis in millimeters
        minor_mm: Minor axis in millimeters
        thickness_mm: Estimated thickness in millimeters
        
    Returns:
        tuple: (volume_cm3, weight_g)
    """
    # Volume of ellipsoid: V = (4/3)πabc
    a = major_mm / 2
    b = minor_mm / 2
    c = thickness_mm / 2

    volume_mm3 = (4/3) * np.pi * a * b * c
    
    # Convert mm³ to cm³
    volume_cm3 = volume_mm3 / 1000.0

    # Potato density ~ 1.05 g/cm³ (average)
    density = 1.05
    weight_g = volume_cm3 * density
    return volume_cm3, weight_g


def annotate_image(img, coin=None, contour=None, measurements=None):
    """
    Draw annotations on image showing detected objects and measurements.
    
    Args:
        img: BGR image to annotate
        coin: (x, y, r) of detected coin
        contour: Potato contour points
        measurements: Dict with measurement values
        
    Returns:
        Annotated BGR image
    """
    vis = img.copy()
    
    # Draw coin
    if coin is not None:
        x, y, r = coin
        cv2.circle(vis, (x, y), r, (0, 255, 255), 3)  # Yellow
        cv2.circle(vis, (x, y), 3, (0, 255, 255), -1)
        cv2.putText(vis, "Coin", (x - 25, y - r - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw potato contour and ellipse
    if contour is not None and len(contour) >= 5:
        # Draw contour
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)  # Green
        
        # Fit and draw ellipse
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (axis1, axis2), angle = ellipse
        
        # Ensure correct major/minor ordering
        major = max(axis1, axis2)
        minor = min(axis1, axis2)
        corrected_ellipse = ((cx, cy), (major, minor), angle)
        
        cv2.ellipse(vis, corrected_ellipse, (0, 165, 255), 3)  # Orange
        cv2.putText(vis, "Potato", (int(cx) - 35, int(cy) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Add measurement text
    if measurements:
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for key, value in measurements.items():
            text = f"{key}: {value}"
            (text_w, text_h), _ = cv2.getTextSize(text, font, 0.7, 2)
            
            # Background rectangle for readability
            cv2.rectangle(vis, (5, y_offset - text_h - 5), 
                         (text_w + 15, y_offset + 5), (0, 0, 0), -1)
            
            cv2.putText(vis, text, (10, y_offset), font, 0.7, 
                       (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 30
    
    return vis


def estimate_potato_size(image_path):
    """
    Main function to estimate potato size from image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Results including measurements and annotated image
    """
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Cannot load image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Detect coin ----
    coin = detect_coin(gray)
    if coin is None:
        print("❌ No coin detected!")
        return None

    coin_x, coin_y, coin_r = coin
    coin_px = coin_r * 2
    mm_per_px = COIN_DIAMETER_MM / coin_px

    # ---- Detect potato ----
    potato_contour = detect_potato_contour(img, coin_position=coin)
    if potato_contour is None:
        print("❌ Potato not detected!")
        return None

    if len(potato_contour) < 5:
        print("❌ Cannot fit ellipse to potato!")
        return None

    # Fit ellipse and ensure correct major/minor ordering
    ellipse = cv2.fitEllipse(potato_contour)
    (_, _), (axis1, axis2), angle = ellipse
    
    # CRITICAL FIX: Ensure major >= minor
    major_px = max(axis1, axis2)
    minor_px = min(axis1, axis2)

    major_mm = major_px * mm_per_px
    minor_mm = minor_px * mm_per_px

    # Estimated thickness = 60% of minor axis (more realistic)
    thickness_mm = 0.6 * minor_mm

    volume_cm3, weight_g = compute_weight(major_mm, minor_mm, thickness_mm)

    # Print results
    print("\n✅ Potato Size Estimation")
    print(f"Coin diameter (px): {coin_px}")
    print(f"Scale: {mm_per_px:.4f} mm/px")
    print(f"Major axis: {major_mm:.1f} mm")
    print(f"Minor axis: {minor_mm:.1f} mm")
    print(f"Thickness (est): {thickness_mm:.1f} mm")
    print(f"Volume: {volume_cm3:.1f} cm³")
    print(f"Weight: {weight_g:.1f} g")

    # Create annotated image
    measurements = {
        "Scale": f"{mm_per_px:.4f} mm/px",
        "Major": f"{major_mm:.1f} mm",
        "Minor": f"{minor_mm:.1f} mm",
        "Volume": f"{volume_cm3:.1f} cm³",
        "Weight": f"{weight_g:.1f} g"
    }
    
    annotated = annotate_image(img, coin=coin, contour=potato_contour, 
                               measurements=measurements)

    return {
        "scale": mm_per_px,
        "major_mm": major_mm,
        "minor_mm": minor_mm,
        "thickness_mm": thickness_mm,
        "volume_cm3": volume_cm3,
        "weight_g": weight_g,
        "annotated_image": annotated,
        "original_image": img
    }


# Run from terminal
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 estimate_size.py <image_path>")
    else:
        result = estimate_potato_size(sys.argv[1])
        
        if result is not None:
            # Show annotated image
            cv2.imshow("Result", result["annotated_image"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save annotated image
            output_path = "potato_measurement_result.jpg"
            cv2.imwrite(output_path, result["annotated_image"])
            print(f"\nAnnotated image saved as: {output_path}")