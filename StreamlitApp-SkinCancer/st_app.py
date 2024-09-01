import streamlit as st
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import multiprocessing

def app():
    st.title("Skin Cancer Detection")
    model = tf.keras.models.load_model("./ModelExperiment/efficientNetB0/")
    file = st.file_uploader("Upload a CSV")
    txt = ''
    if file:
        st.image(file)
        img = mpimg.imread(file)
        img = np.array([img])
        predictionProb = model(img)
        prediction = int(np.array(tf.squeeze(tf.round(predictionProb))))
        if prediction == 0:
            txt = 'Benign'
        else:
            txt = "Malignant"
    st.header(txt)

if __name__ == '__main__':
   multiprocessing.Process(target=app())