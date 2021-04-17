# import required libraries
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
# load the model
model = tf.keras.models.load_model("sortit_model(v4).hdf5")
# header
st.write(
    """
    # SortIt Image Classifier
    """

)
st.write(
    """
    This app classifies images of cats, cars, dogs, bicycles, and motorcycles
    """
)
# upload an image for prediction
file = st.file_uploader("Please upload an image file", type=["jpg","png"])

# function that imports image resizes it, and runs prediction
def import_and_predict(img_data,model):
    # specify the image size
    size = (150,150)
    # import the image and resize it
    image = ImageOps.fit(img_data,size,Image.ANTIALIAS)
    # convert image to numpy array
    image = np.asarray(image)
    # convert the image from bgr to rgb
    # img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # resize the numpy array
    img_resize = (cv2.resize(image,dsize=(75,75),interpolation=cv2.INTER_CUBIC))/255
    # reshape the image
    img_reshape = img_resize[np.newaxis,...]
    # predictions = np.argmax(model.predict(img_reshape), axis=-1)
    # run prediction on the image
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    #open the image and convert it to rgb
    image = Image.open(file).convert('RGB')
    st.image(image,use_column_width=True)
    prediction = import_and_predict(image,model)
    # print the prediction
    st.write(prediction)