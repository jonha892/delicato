import numpy as np
import cv2
from PIL import Image

def preprocess_image(image_pth):
    
    # Convert image to grayscale
    gray = image_pth.convert("L")

    # Convert grayscale image to numpy array
    img = np.array(gray)
    # Apply median blur
    blur = cv2.medianBlur(img,3)

    # Define kernel for morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Perform erosion
    erode = cv2.erode(blur, kernel, iterations=2)

    # Perform dilation
    dilate = cv2.dilate(erode, kernel, iterations=1)

    # Apply thresholding
    _, binary = cv2.threshold(dilate, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find the bounding box coordinates of the non-white pixels
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)

    # Add extra white space to the bounding box coordinates
    padding = 20  # Adjust the padding size as needed
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Make sure the coordinates are within the image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    # Crop the image using the modified bounding box coordinates
    cropped_image = binary[y:y+h, x:x+w]

    # Add extra white space around the cropped image
    extra_space = np.zeros((cropped_image.shape[0] + 2 * padding, cropped_image.shape[1] + 2 * padding), dtype=np.uint8) * 255
    extra_space[padding:-padding, padding:-padding] = cropped_image
    
    # Convert the numpy array back to PIL image
    resized_image = Image.fromarray(extra_space)

    return resized_image