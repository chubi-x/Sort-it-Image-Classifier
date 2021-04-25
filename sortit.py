# import required libraries
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
# load the model
model = tf.keras.models.load_model( "sortit_model(v4).hdf5" )
# header
st.title(
    """
    SortIt Image Classifier
    """

)
st.write(
    """
    This app classifies images of cats, cars, dogs, bicycles, and motorcycles
    """
)


# function that imports image resizes it, and runs prediction
def import_and_predict(img_data, model):
    # specify the image size
    size = (75, 75)
    # import the image and resize it
    image = ImageOps.fit( img_data, size, Image.ANTIALIAS )
    # convert image to numpy array
    image = np.asarray( image )
    # convert the image from bgr to rgb
    # img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # resize the numpy array
    # img_resize = (cv2.resize( image, dsize=(75, 75), interpolation=cv2.INTER_CUBIC )) / 255
    img_resize = image / 255
    # reshape the image numpy array
    img_reshape = img_resize[np.newaxis, ...]
    # predictions = np.argmax(model.predict(img_reshape), axis=-1)
    # run prediction on the image
    prediction = model.predict( img_reshape )
    return prediction


# upload an image for prediction
file = st.file_uploader( "Please upload an image file", type=["jpeg", "png","jpg"] )

if file is None:
    st.text( "Please upload an image file" )
else:
    # open the image and convert it to rgb
    image = Image.open( file ).convert( 'RGB' )
    # display the image
    st.image( image, use_column_width="auto")
    # run the prediction
    prediction = import_and_predict( image, model )
    pred = prediction[0]
    # get the class with the highest probability
    max_pred = np.amax( pred )
    # check if the highest probability is greater than 50%
    if max_pred > 0.5:
        if max_pred == pred[0]:
            st.success( "What did the cycle say to his parents? I'm bi" )
            st.balloons()
        elif max_pred == pred[1]:
            st.success( "VROOM VROOM cool whip dude" )
            st.balloons()
        elif max_pred == pred[2]:
            st.success( "Cute Kitty say meowww" )
            st.balloons()
        elif max_pred == pred[3]:
            st.success( "Did I hear a WOOF? cause this looks like a cute DOGE" )
            st.balloons()
        elif max_pred == pred[4]:
            st.success( "nice bike dude" )
            st.balloons()
    else:
        st.write("Idk what this is. try uploading an image of a bicycle, car, cat, dog, or motorcyle")
