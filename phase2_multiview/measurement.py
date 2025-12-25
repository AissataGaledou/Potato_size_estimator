import cv2

# Calibrated based on actual potato measurements
# Rule: MM_PER_PIXEL = actual_size_mm / measured_pixels
MM_PER_PIXEL = 0.1  # Changed from 0.5 to 0.1

def contour_dimensions(contour):
    x, y, w, h = cv2.boundingRect(contour)
    width_mm = w * MM_PER_PIXEL
    height_mm = h * MM_PER_PIXEL
    return width_mm, height_mm