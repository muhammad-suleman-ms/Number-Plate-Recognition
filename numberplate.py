import cv2
import os
import pytesseract

# Set the path to the Haar cascade for detecting license plates
harcascade = 'C:\PROJECT UNIVERSITY\pak.xml'

# Set the path to the folder where the license plate images will be saved
save_folder = 'images'

# Create the save folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Open the video file (replace '0' with the desired camera index)
cap = cv2.VideoCapture(0)

# Open the video file (replace '0' with the desired camera index)
cap = cv2.VideoCapture(0)


# Set the path to the pytesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

while cap.isOpened():
    # Read a frame from the camera
    success, img = cap.read()

    if not success:
        break

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade for license plate detection
    plate_cascade = cv2.CascadeClassifier(harcascade)

    # Detect license plates in the image
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Process each detected license plate
    for (x, y, w, h) in plates:
        area = w * h

        # Check if the area of the detected region exceeds the minimum area threshold
        if area > 500:
            # Increase the size of the ROI
            x_roi = max(0, x - 20)  # Adjusted x-coordinate for ROI
            y_roi = max(0, y - 20)  # Adjusted y-coordinate for ROI
            w_roi = min(img.shape[1] - x_roi, w + 40)  # Adjusted width for ROI
            h_roi = min(img.shape[0] - y_roi, h + 40)  # Adjusted height for ROI

            # Draw a green rectangle around the license plate
            cv2.rectangle(img, (x_roi, y_roi), (x_roi+w_roi, y_roi+h_roi), (0, 255, 0), 2)

            # Extract the region of interest (ROI) containing the license plate
            img_roi = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

            # Perform OCR on the ROI to extract the license plate number
            text = pytesseract.image_to_string(img_roi, config='--psm 10 --oem 3')

            # Extract only alphanumeric characters
            filtered_text = ''.join(e for e in text if e.isalnum())

            # Create a text file with the extracted text as the filename
            file_name = f'{save_folder}/{filtered_text}.txt'
            with open(file_name, 'w') as f:
                f.write(text)

            # Display the ROI
            cv2.imshow('ROI', img_roi)

    # Display the result
    cv2.imshow('Result', img)

    # Check for the 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
