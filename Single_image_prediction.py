# Convolutional Neural Network



# Installing numpy
# pip install numpy

# Installing Keras
# pip install --upgrade keras






# ADD THIS SECTION OF CODE BELOW IN GENDER_CLASSIFICATION FILE

# AFTER TRAINING THE CLASSIFIER OR MODEL
 


# Making new predictions


import numpy as np

from keras.preprocessing import image


# Loading the image

# CHANGE THE IMAGE_NAME WITH YOUR'S IMAGE

test_image = image.load_img('dataset/predict_image/image_name.jpg', target_size = (64, 64))

# Converting image to array of pixels

test_image = image.img_to_array(test_image)

# Expanding dimension

test_image = np.expand_dims(test_image, axis = 0)

# Predict the image

result = classifier.predict(test_image)

# Checking the prediction

training_set.class_indices


if result[0][0] == 1:
    prediction = 'male'
    
else:
    prediction = 'female'
