import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

model = load_model("C:\\MyFiles\\Project\\AI camera\\cnnmodel.h5")
# print(model.summary())

def img_aug(img):
    test_image = cv2.resize(img, (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    return test_image

def prediction(test_image):
    result = model.predict(test_image)
    print("dog" if result[0][0] == 1 else "cat")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame
    test_image = img_aug(frame)
    
    # Make prediction
    pred = prediction(test_image)
    
    # Display the resulting frame
    cv2.putText(frame, pred, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Live Feed', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()