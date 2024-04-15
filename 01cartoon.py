import cv2
import numpy as np
from tkinter.filedialog import *

photo=askopenfilename()
img= cv2.imread(photo)  #reads image file from the path

grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #BGR to Grayscale
grey=cv2.medianBlur(grey,5)   #remove noise (replace pixels value )
edges=cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,9)  #create binary images based on pixel intensity

color=cv2.bilateralFilter(img, 9, 250, 250)  #smooth the image by preserving the edges
cartoon= cv2.bitwise_and(color, color, mask= edges)  #performs bitwise AND (the adaptive treashold mask to color image)

cv2.imshow("Image",img)
cv2.imshow("Cartoon", cartoon)

cv2.imwrite("cartoon.jpg", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()