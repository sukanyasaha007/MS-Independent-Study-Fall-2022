import cv2
 
# path
path = r"images\track1.jpg"
 
# Reading an image in default mode
src = cv2.imread(path)
 
# Window name in which image is displayed
window_name = 'Image'
 
# Using cv2.flip() method
# Use Flip code 0 to flip vertically
image = cv2.flip(src, 2)

# Displaying the image
# cv2.imshow(window_name, image)
# cv2.waitKey(0)

cv2.imwrite(r'images\track1_flipped.PNG', image)
