
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt

from PIL import Image
from pathlib import Path

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Road Condition Monitoring",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ Deep Learning Based Road Condition Monitoring")

st.write(
    "Upload a road image to classify its condition "
    "using three deep learning models."
)

# -----------------------------------
# CONFIGURATION
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

IMG_SIZE = 224

CLASS_NAMES = ["Smooth", "Crack", "Pothole"]

RISK_MAP = {
    "Smooth": "Low Risk",
    "Crack": "Medium Risk",
    "Pothole": "High Risk"
}

MODEL_PATHS = {
    "Custom CNN": MODEL_DIR / "Custom_CNN.keras",
    "ResNet50": MODEL_DIR / "ResNet50.keras",
    "EfficientNetB0": MODEL_DIR / "EfficientNetB0.keras"
}

# Test-set accuracies from our experiments
TEST_ACCURACY = {
    "Custom CNN": 84.88,
    "ResNet50": 97.67,
    "EfficientNetB0": 95.35
}

# -----------------------------------
# LOAD MODELS
# -----------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, path in MODEL_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded_models[name] = tf.keras.models.load_model(
            path,
            compile=False
        )

    return loaded_models


# -----------------------------------
# IMAGE PREPROCESSING
# -----------------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# -----------------------------------
# PREDICTION
# -----------------------------------

def predict_all_models(models, image_array):

    predictions = {}

    for name, model in models.items():

        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predictions[name] = {
            "class": CLASS_NAMES[predicted_index],
            "probabilities": probabilities,
            "score": float(
                probabilities[predicted_index]
            ) * 100
        }

    return predictions


# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Uploaded Road Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Image Information")

        st.write("Filename:", uploaded_file.name)
        st.write("Original dimensions:", image.size)
        st.write("Model input:", "224 × 224 × 3")

    if st.button(
        "Analyze Road Condition",
        type="primary"
    ):

        try:

            # -----------------------------------
            # LOAD MODELS
            # -----------------------------------

            with st.spinner(
                "Loading deep learning models..."
            ):

                models = load_models()

            # -----------------------------------
            # ANALYZE IMAGE
            # -----------------------------------

            with st.spinner(
                "Analyzing road image..."
            ):

                image_array = preprocess_image(image)

                predictions = predict_all_models(
                    models,
                    image_array
                )

            st.success("Analysis completed!")

            # -----------------------------------
            # INDIVIDUAL MODEL PREDICTIONS
            # -----------------------------------

            st.header("Model Predictions")

            columns = st.columns(3)

            for column, (name, result) in zip(
                columns,
                predictions.items()
            ):

                with column:

                    st.subheader(name)

                    st.metric(
                        "Predicted Condition",
                        result["class"]
                    )

                    st.metric(
                        "Prediction Score",
                        f"{result['score']:.2f}%"
                    )

                    st.caption(
                        f"Test accuracy: "
                        f"{TEST_ACCURACY[name]:.2f}%"
                    )

            # -----------------------------------
            # GROUPED PROBABILITY COMPARISON
            # -----------------------------------

            st.header("Class Probability Comparison")

            probability_data = {}

            for name, result in predictions.items():

                probability_data[name] = (
                    result["probabilities"] * 100
                )

            probability_df = pd.DataFrame(
                probability_data,
                index=CLASS_NAMES
            )

            # Convert wide data into long format
            chart_df = (
                probability_df
                .rename_axis("Class")
                .reset_index()
                .melt(
                    id_vars="Class",
                    var_name="Model",
                    value_name="Probability"
                )
            )

            # Grouped bar chart
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(

                    x=alt.X(
                        "Class:N",
                        sort=CLASS_NAMES,
                        title="Road Condition"
                    ),

                    xOffset=alt.XOffset(
                        "Model:N"
                    ),

                    y=alt.Y(
                        "Probability:Q",
                        title="Prediction Score (%)",
                        scale=alt.Scale(
                            domain=[0, 100],
                            nice=False
                        )
                    ),

                    color=alt.Color(
                        "Model:N",
                        title="Deep Learning Model"
                    ),

                    tooltip=[
                        alt.Tooltip(
                            "Class:N",
                            title="Road Condition"
                        ),

                        alt.Tooltip(
                            "Model:N",
                            title="Model"
                        ),

                        alt.Tooltip(
                            "Probability:Q",
                            title="Prediction Score",
                            format=".2f"
                        )
                    ]
                )
                .properties(
                    height=420
                )
            )

            st.altair_chart(
                chart,
                use_container_width=True
            )

            # -----------------------------------
            # PROBABILITY TABLE
            # -----------------------------------

            st.subheader("Probability Details")

            st.dataframe(
                probability_df.round(2),
                use_container_width=True
            )

            # -----------------------------------
            # BEST MODEL PREDICTION
            # -----------------------------------

            best_model = max(
                TEST_ACCURACY,
                key=TEST_ACCURACY.get
            )

            best_result = predictions[best_model]

            final_class = best_result["class"]

            risk = RISK_MAP[final_class]

            # -----------------------------------
            # FINAL ROAD CONDITION ASSESSMENT
            # -----------------------------------

            st.header("Final Road Condition Assessment")

            result_col1, result_col2, result_col3 = st.columns(3)

            result_col1.metric(
                "Selected Model",
                best_model
            )

            result_col2.metric(
                "Road Condition",
                final_class
            )

            result_col3.metric(
                "Risk Level",
                risk
            )

            st.info(
                f"The selected model, {best_model}, "
                f"predicted {final_class} with a "
                f"prediction score of "
                f"{best_result['score']:.2f}%."
            )

            st.caption(
                "Risk level is assigned using the project's "
                "class-to-risk mapping. Prediction scores are "
                "model softmax outputs and are not calibrated "
                "probabilities of real-world safety."
            )

        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )

else:

    st.info(
        "Please upload a road image to begin analysis."
    )


# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "Deep Learning Based Road Condition Monitoring "
    "| Custom CNN | ResNet50 | EfficientNetB0"
)
