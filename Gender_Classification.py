# Convolutional Neural Network

# Installing Theano
# pip install theano

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()
    
# Adding first Convolution layer
classifier.add(Conv2D(32, (3, 3), padding = "same", input_shape = (64, 64, 3), activation = 'relu'))
    
# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
# Adding second convolutional layer
classifier.add(Conv2D(32, (3, 3), padding = "same", activation = 'relu'))
    
# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding third convolutional layer
classifier.add(Conv2D(32, (3, 3), padding = "same", activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding fourth convolutional layer
classifier.add(Conv2D(32, (3, 3), padding = "same", activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

    
# Flattening
classifier.add(Flatten())
    
# Fully connection layers
classifier.add(Dense(units = 64, activation = 'relu'))
    
# Adding Dropout
classifier.add(Dropout(0.6))

classifier.add(Dense(units = 64, activation='relu'))

classifier.add(Dense(units = 64, activation='relu'))

# Adding Dropout
classifier.add(Dropout(0.3))
    
# Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
# Compiling the CNN    
optimizer = Adam(lr=1e-3)
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)
