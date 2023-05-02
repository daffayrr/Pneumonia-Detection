import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the Keras model
model = tf.keras.models.load_model('model.h5')

# Define the labels for the object classes
labels = ['NORMAL', 'PNEUMONIA']

# Define a function to make predictions on an image
def predict(image):
    # Preprocess the image
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0) / 255.0

    # Make a prediction
    predictions = model.predict(image_array)
    label_index = tf.argmax(predictions, axis=1).numpy()[0]
    label = labels[label_index]

    return label

# Define the Streamlit app
st.set_page_config(page_title='Pneumonia Detection', page_icon=':detective:', layout='wide')
st.markdown("""<style>.stApp { <!--max-width: 800px;--> margin:ceenter;}</style>""", unsafe_allow_html=True)

st.markdown("""<div class="container-fluid"><div class="row"><div class="col-md-8"><h1 class="display-4">Pneumonia Detection</h1></div></div></div>""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    # Allow the user to upload an image
    st.markdown('<h4>Upload Gambar</h4>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])

with col2:
    if uploaded_file is not None:
        # Make a prediction and display the result
        image = Image.open(uploaded_file)
        label = predict(image)
        st.markdown(f'<h4>Prediksi: {label}</h4>', unsafe_allow_html=True)
        st.image(image, use_column_width=True)

st.markdown("""<hr class="my-4"><div class="container-fluid"><div class="row"><div class="col-md-12"><p class="lead">&copy; 2023, Farras Daffa</p></div></div></div>""", unsafe_allow_html=True)
