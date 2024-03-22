import cv2

# Read the image
image = cv2.imread("cat.jpg")  # Replace "your_image.jpg" with the path to your image file

# Resize the image to 32x32 pixels
resized_image = cv2.resize(image, (32, 32))

# Save the resized image
cv2.imwrite("resized_cat.jpg", resized_image)  # Replace "resized_image.jpg" with the desired output file path