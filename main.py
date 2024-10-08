import tensorflow as tf
import cv2 as cv
import os
import pytesseract

path = 'D:\Coding\Follow_Detection\Sample'

file_path = r"D:\Coding\Follow_Detection\Sample\train\ef2dc97f-bc6d-49a3-9ddb-03f54f3a20e7___Skoda-Rapid-new-Exterior-84398.jpg.jpeg"
def display_img(img):
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def feature_extraction(file_path):
    image = cv.imread(file_path)
    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Use bilateral filter to reduce noise
    blurred_image = cv.bilateralFilter(gray_image, 11, 17, 17)

    # Apply edge detection
    edged_image = cv.Canny(blurred_image, 30, 200)
    

    # Find contours based on edges
    contours, _ = cv.findContours(edged_image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    # Initialize the contour that represents the number plate
    number_plate_contour = None

    # Loop over the contours to find a potential license plate area (usually rectangular)
    for contour in contours:
        # Approximate the contour to reduce the number of points
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
        # If the contour has 4 corners, we assume it's a number plate
        if len(approx) == 4:
            number_plate_contour = approx
            break
    if number_plate_contour is not None:
        # Create a mask for the number plate
        mask = cv.drawContours(image.copy(), [number_plate_contour], -1, (0, 255, 0), 3)

        # Crop the number plate
        x, y, w, h = cv.boundingRect(number_plate_contour)
        number_plate_image = image[y:y+h, x:x+w]
    else:
        print("Number plate not found.")
    display_img(number_plate_image)
    # Convert the cropped image to grayscale
    number_plate_gray = cv.cvtColor(number_plate_image, cv.COLOR_BGR2GRAY)

    # Use OCR to recognize text from the number plate
    text = pytesseract.image_to_string(number_plate_gray, config='--psm 8')
    print("Detected Number Plate:", text)


feature_extraction(file_path)

# for root, dir, files in os.walk(path):
#     for file in files:
#         if file.endswith('.jpeg'):
#             img_path = os.path.join(root, file)
#             img = cv.imread(img_path, cv.IMREAD_COLOR)
#             cv.imshow(file, img)
#             cv.waitKey(0)
#             cv.destroyAllWindows()

        