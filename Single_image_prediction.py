

# Requirements

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

test_image_1 = image.load_img('dataset/predict_image/male_female1.jpg', target_size = (64, 64))

test_image_2 = image.load_img('dataset/predict_image/male_female2.jpg', target_size = (64, 64))

# Converting image to array of pixels

test_image_1 = image.img_to_array(test_image_1)

test_image_2 = image.img_to_array(test_image_2)


# Expanding dimension

test_image_1 = np.expand_dims(test_image_1, axis = 0)

test_image_2 = np.expand_dims(test_image_2, axis = 0)

# Predict the image

result = classifier.predict(test_image_1)

# Checking the prediction

print(training_set.class_indices)


if result[0][0] == 1:
    prediction = 'male'
    
else:
    prediction = 'female'
  
  
print(result)  

# Predict the image

result = classifier.predict(test_image_2)

if result[0][0] == 1:
    prediction = 'male'
    
else:
    prediction = 'female'
  

print(result)  
