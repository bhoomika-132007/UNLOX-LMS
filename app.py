
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Bhoomika | Digit Recognition",
    page_icon="🔢"
)

MODEL_PATH = "mnist_cnn_bhoomika.keras"

@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model(MODEL_PATH)

def prepare_image(uploaded_image):
    image = Image.open(uploaded_image).convert("L")
    image = ImageOps.autocontrast(image)
    image.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), color=0)
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))

    array = np.asarray(canvas).astype("float32") / 255.0

    if array.mean() > 0.5:
        array = 1.0 - array

    return array[np.newaxis, ..., np.newaxis], canvas

st.title("🔢 Handwritten Digit Recognition")
st.caption("CNN Image Classification | Student Project by Bhoomika")

st.write("Upload a handwritten digit image (0–9) to predict its class.")

uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        input_array, preview = prepare_image(uploaded_file)
        st.image(preview, caption="Preprocessed image", width=180)

        with st.spinner("Analysing image..."):
            model = load_digit_model()
            scores = model.predict(input_array, verbose=0)[0]

        predicted_digit = int(np.argmax(scores))
        confidence = float(scores[predicted_digit])

        st.success(f"Predicted digit: {predicted_digit}")
        st.metric("Prediction confidence", f"{confidence:.2%}")

        st.subheader("Class probabilities")
        st.bar_chart({str(i): float(scores[i]) for i in range(10)})

    except Exception as error:
        st.error(f"Error: {error}")
else:
    st.info("Upload a digit image to see the prediction.")

with st.expander("About this project"):
    st.write(
        "This application uses a CNN trained on the MNIST "
        "handwritten digit dataset."
    )
