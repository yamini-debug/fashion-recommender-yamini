import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="✨",
    layout="wide"
)

# ---------------- LOAD FILES ----------------

feature_list = np.array(
    pickle.load(open('embeddings.pkl', 'rb'))
)

filenames = pickle.load(
    open('filenames.pkl', 'rb')
)

# WINDOWS PATH -> LINUX PATH FIX

filenames = [file.replace("\\", "/") for file in filenames]

# ---------------- MODEL ----------------

@st.cache_resource
def load_model():

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = tensorflow.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    return model

model = load_model()

# ---------------- NEIGHBORS ----------------

neighbors = NearestNeighbors(
    n_neighbors=6,
    algorithm='brute',
    metric='euclidean'
)

neighbors.fit(feature_list)

# ---------------- HEADER ----------------

st.title("Fashion Recommendation System")
st.write("Upload an image and get similar fashion recommendations.")

# ---------------- SAVE FILE ----------------

def save_uploaded_file(uploaded_file):

    try:

        os.makedirs('uploads', exist_ok=True)

        file_path = os.path.join(
            'uploads',
            uploaded_file.name
        )

        with open(file_path, 'wb') as f:

            f.write(uploaded_file.getbuffer())

        return file_path

    except Exception as e:

        st.error(f"Error Saving File: {e}")

        return None

# ---------------- FEATURE EXTRACTION ----------------

def feature_extraction(img_path, model):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    expanded_img_array = np.expand_dims(
        img_array,
        axis=0
    )

    preprocessed_img = preprocess_input(
        expanded_img_array
    )

    result = model.predict(
        preprocessed_img,
        verbose=0
    ).flatten()

    normalized_result = result / norm(result)

    return normalized_result

# ---------------- RECOMMEND ----------------

def recommend(features):

    distances, indices = neighbors.kneighbors([features])

    return indices

# ---------------- FILE UPLOADER ----------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=['png', 'jpg', 'jpeg']
)

# ---------------- MAIN ----------------

if uploaded_file is not None:

    saved_path = save_uploaded_file(uploaded_file)

    if saved_path is not None:

        display_image = Image.open(uploaded_file)

        st.image(display_image, width='stretch')

        with st.spinner('Finding similar products...'):

            features = feature_extraction(
                saved_path,
                model
            )

            indices = recommend(features)

        st.success('Recommendations generated successfully!')

        cols = st.columns(5)

        for i in range(5):

            index = indices[0][i + 1]

            image_path = filenames[index]

            # EXTRA SAFETY CHECK

            if os.path.exists(image_path):

                with cols[i]:

                    st.image(
                        image_path,
                        width='stretch'
                    )

            else:

                with cols[i]:

                    st.write("Image Missing")

    else:

        st.error("File upload failed.")