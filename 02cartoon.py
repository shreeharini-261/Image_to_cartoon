import cv2
import numpy as np

img = cv2.imread('pic2.png')

if img is None:
    print("Error: Could not read image 'pic1.jpeg'. Please check the file path.")
    exit()

def cartoonize(image, num_clusters):
   
    data = np.float32(image.reshape((-1, 3)))

    print("Shape of original image:", image.shape)
    print("Shape of reshaped data:", data.shape)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    _, label, center = cv2.kmeans(data, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    print("K-Means cluster centers:\n", center)

    result = center[label.flatten()]
    result = result.reshape(image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

    print("Edges image shape:", edges.shape)
    print("Edges image type:", edges.dtype)
    
    blurred = cv2.medianBlur(result, 3)

    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    return cartoon  


cartoonized_image = cartoonize(img.copy(), 8)  

if cartoonized_image is None:
    print("Error: Cartoonized image is None. Please check the cartoonize function.")
else:
  
    print("Cartoonized image shape:", cartoonized_image.shape)
    print("Cartoonized image type:", cartoonized_image.dtype)

    cv2.imshow("Cartoonized Image", cartoonized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
