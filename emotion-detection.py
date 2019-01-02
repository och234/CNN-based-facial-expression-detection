import glob #used for iterating through the folders
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import layer_utils
import keras.backend as K

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list

data = {}
def get_files(emotion):
'''matches all the pathname in the dataset folder that matches with emotion'''
    files = glob.glob("dataset\\%s\\*" %emotion) 
    random.shuffle(files) #shuffle the files randomly to prevent bias
    train_pred = files
    return train_pred
def make_sets():
    train_data = []
    train_labels = []
    for emotion in emotions:
	#iterate through the emotions list to get the files in each folder
        train_pred = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in train_pred:
            image = plt.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            gray = gray.T #get the transpose of the array
            gray = np.expand_dims(gray, axis=0).T #add a one to the array and transpose back
            
            train_data.append(gray) #append image array to training data list
            train_labels.append(emotions.index(emotion))
    return train_data, train_labels

train, label = make_sets()

'''convert the train and label set to an array'''
a_train = np.asarray(train)
a_label = np.asarray(label)

'''Share into train and test set'''
train_set, test_set, train_label, test_label = train_test_split(a_train, a_label, test_size=0.25)

'''Confirming we are on the right track'''
print("the number of training data is", train_set.shape[0]) #output 476
print("the number of test data is", test_set.shape[0]) #output 159
print("the shape of training data is", train_set.shape) #output (476, 350, 350, 1)
print("the shape of prediction data is", test_set.shape) #output (159, 350, 350, 1)

#normalizing the features
train_set = train_set/255
test_set = test_set/255

#turing the labels to one hot vectors
train_label = np_utils.to_categorical(train_label, num_classes=8)
test_label = np_utils.to_categorical(test_label, num_classes=8)


def EmotionModel(shape):
'''we use a 7 layer model '''
    
    input_image = Input(shape)
    X = ZeroPadding2D((3, 3))(input_image) #using a padding of 3 to achieve same padding
    X = Conv2D(32, (7, 7), strides=(1,1))(X) #using 32 filters of size 7*7 and strides of 1
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X) #using relu activation function
    X = Conv2D(64, (7, 7), strides=(1,1), name='conv0')(X)#using 64 filters of size 7*7 and strides of 1    
    
    X = Conv2D(128, (5, 5), strides=(1,1), name='conv2')(X) #using 128 filters of size 5*5 and strides of 1
    
    X = MaxPooling2D((2, 2), name='max_pool1')(X)#using max pooling with a filter size of 2 * 2
     
    X = Flatten()(X) #converting the CNN model to fully-connected model
    
	'''The rest part is just a normal neural network'''
    X = Dense(256, activation='relu', name='fc1')(X) 
    X = Dense(64, activation='relu', name='fc2')(X)
    X = Dense(8, activation='softmax', name='fc3')(X)
    
    model = Model(inputs=input_image, outputs=X, name='EmotionModel')

    return model

	
emotionModel = EmotionModel(np.asarray([350,350,1])) #adding a 1 because we are expecting a 3 dimensional value, the one doesn't affect the accuracy as it was added in both training and test set

emotionModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

'''with just 7 epoch we were able to achieve a test accuracy of above 85% and 87 at times'''
emotionModel.fit(train_set, train_label, epochs=10, batch_size=32)

preds = emotionModel.evaluate(test_set, test_label, batch_size=32, verbose=1, sample_weight=None)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))