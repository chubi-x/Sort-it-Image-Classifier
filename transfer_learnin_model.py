# import all the required packages
import sys

import sklearn
import tensorflow as tf
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import save_model
import numpy as np
import seaborn as sns
import pandas as pd

# specify the image size for the images
img_size = 75
# load the train and testing image data from numpy arrays
train = np.load('train-data.npy', allow_pickle=True)
test = np.load('test-data.npy', allow_pickle=True)

# visualize the training data
training_data = []
# loop that saves the labels of each class into an array
for i in train:
    if i[1] == 0:
        training_data.append("bicycle")
    elif i[1] == 1:
        training_data.append('car')
    elif i[1] == 2:
        training_data.append('cat')
    elif i[1] == 3:
        training_data.append('dog')
    elif i[1] == 4:
        training_data.append('motorcycle')
# set the plot style
sns.set_style('darkgrid')
# visualize the training data
plot = sns.countplot(training_data)
# save the plot to a file
plot.figure.savefig('training-data.png')

# create arrays that hold the training features and labels
x_train = []
y_train = []
# create arrays that hold the testing feature and labels
x_val = []
y_val = []
# append the features and values to their respective arrays
for feature, label in train:
    x_train.append(feature)
    y_train.append(label)
# append the features and values to their respective arrays
for feature, label in test:
    x_val.append(feature)
    y_val.append(label)
# Normalize the data
# convert training feature to numpy array
x_train = np.asarray(x_train).astype(np.float32)
# resize the array
x_train = x_train/255
x_train.reshape(-1, img_size, img_size, 1)

# convert testing features to numpy array
x_val = np.asarray(x_val).astype(np.float32)
# resize the array
x_val = x_val/255
x_val.reshape(-1, img_size, img_size, 1)
# convert training and testing labels to numpy array
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
# convert the training and testing labels to binary class matrix to obtain classification report
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=5)

# initialize training data generator with image augmentation applied
train_data_generator = ImageDataGenerator(rotation_range=30,
                                          zoom_range=0.3, width_shift_range=0.2,
                                          height_shift_range=0.2, validation_split=0.15, shear_range=0.2,
                                          horizontal_flip=True)
# fit the data generator on the training features
train_data_generator.fit(x_train)

# function that plots model history
def summarize_diagnostics(history):
    # first plot the loss
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='red', label='test')
    # then plot the accuracy
    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='red', label='test')
    #  save the plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


# create pre trained InceptionV3 model with imagenet weights. remove the final layer and retrain it to classify 
# images of bicycles, cars, cats, dogs, and motorycycles
pre_trained_model = InceptionV3(input_shape=(75, 75, 3),
                                include_top=False,
                                weights='imagenet')
# # make the final layer mixed5
final_layer = pre_trained_model.get_layer('mixed5')
# make the final output the output from the final layer
final_output = final_layer.output
# convert the model to a sequential model with Dropout added
transfer_learning_model = tf.keras.Sequential([pre_trained_model,
                                               Flatten(),
                                               Dropout(0.4),
                                               Dense(5, activation="softmax")])
# compile the model
transfer_learning_model.compile(optimizer='adam',
                                loss='categorical_crossentropy', metrics=['accuracy'])


# create a callback to stop training when validation accuracy is >= 90%
class callBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') >= 0.90:
            print("validation accuracy is ", logs.get('val_accuracy'))
            print("\n Reached desired validation accuracy. stopping training")
            # stop model training
            self.model.stop_training = True


# bind callback to a variable
callback = callBack()
# fit the model on the training data
transfer_learning_history = transfer_learning_model.fit(
    x_train, y_train,
    steps_per_epoch=120,
    epochs=4,
    verbose=1,
    validation_data=(x_val, y_val),
    callbacks=[callback]
)
# get the accuracy history
summarize_diagnostics(transfer_learning_history)
# make predictions
predictions = np.argmax(transfer_learning_model.predict(x_val), axis=-1)
predictions = predictions.reshape(1, -1)[0]
# convert labels back to single digits
y_val = np.argmax(y_val, axis=1)
y_val = np.argmax(y_val, axis=1)
print(classification_report(y_val, predictions, target_names=['Bicycle (Class 0)', 'Car (Class 1)', 'Cats (Class 2)',
                                                              'Dogs (Class 3)', 'Motorcycle (Class 4)']))
# print confusion matrix
confusion_matrix = confusion_matrix(y_val, predictions)
# convert confusion matrix to a dataframe
cm_df = pd.DataFrame(confusion_matrix, range(5), range(5))
# plot the confusion matrix
sns.heatmap(cm_df, annot=True, annot_kws={"size": 16}).figure.savefig("confusion-matrix.png")
pyplot.show()
# save the model 
save_model(transfer_learning_model, 'sortit_model(v4).hdf5')