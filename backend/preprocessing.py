"""
LedgerPilot - Phase 2: Image Pre-processing (OpenCV)


1. Grayscale       — Tesseract works on single-channel intensity, not color.
2. Denoise         — phone photos have sensor noise; denoising BEFORE
                      thresholding prevents noise from becoming false "text".
3. Auto-crop        — finds the document's contour and perspective-corrects it,
                      so background clutter (table, hand, shadow) doesn't
                      confuse OCR. This is the step that turns a "photo of a
                      receipt on a table" into "just the receipt".
4. Deskew          — corrects rotation (a crooked phone angle) using the
                      minimum-area bounding rectangle of the text mass.
5. Adaptive threshold — converts to clean black-on-white, which is what
                      Tesseract is tuned for; adaptive (not global) threshold
                      handles uneven lighting across a phone photo.

"""

import cv2
import numpy as np


def to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(gray_image):
    return cv2.fastNlMeansDenoising(gray_image, h=10, templateWindowSize=7, searchWindowSize=21)


def _order_points(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def auto_crop_document(original_image, gray_image):
    """Finds the largest 4-sided contour (assumed to be the document edge)
    and applies a perspective transform to get a top-down, cropped view.
    Falls back to the original image untouched if no clear document
    boundary is found (e.g. the photo is already a tight crop / a clean
    scan) — better to skip cropping than to crop wrong."""
    edged = cv2.Canny(gray_image, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return original_image, False

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    doc_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * original_image.shape[0] * original_image.shape[1]:
            doc_contour = approx
            break

    if doc_contour is None:
        return original_image, False

    pts = doc_contour.reshape(4, 2).astype("float32")
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 10 or max_height < 10:
        return original_image, False

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(original_image, m, (max_width, max_height))
    return warped, True


def deskew(gray_image):
    """Estimates rotation angle from the text mass's minimum-area bounding
    rectangle and rotates to correct it. Handles typical phone-photo tilt
    (a few degrees), not extreme rotation."""
    thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return gray_image, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return gray_image, angle  # not worth rotating for sub-degree noise

    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray_image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def adaptive_binarize(gray_image):
    return cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=15,
    )


def preprocess_pipeline(image_path, save_intermediate_dir=None):
    """Runs the full pipeline and returns the final binarized image ready
    for OCR. If save_intermediate_dir is given, writes each stage's output
    there (stage1_gray.png, stage2_denoised.png, ...) — use this for your
    ISE 1 demo slides."""
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    stages = {}

    gray = to_grayscale(original)
    stages["stage1_gray"] = gray

    denoised = denoise(gray)
    stages["stage2_denoised"] = denoised

    cropped_color, was_cropped = auto_crop_document(original, denoised)
    cropped_gray = to_grayscale(cropped_color)
    stages["stage3_cropped"] = cropped_gray

    deskewed, angle = deskew(cropped_gray)
    stages["stage4_deskewed"] = deskewed

    final = adaptive_binarize(deskewed)
    stages["stage5_binarized"] = final

    if save_intermediate_dir:
        import os
        os.makedirs(save_intermediate_dir, exist_ok=True)
        for name, img in stages.items():
            cv2.imwrite(os.path.join(save_intermediate_dir, f"{name}.png"), img)

    return final, {"was_cropped": was_cropped, "deskew_angle_deg": round(float(angle), 2)}
