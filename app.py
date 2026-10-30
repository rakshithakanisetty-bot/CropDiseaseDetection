import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from gtts import gTTS
import io

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CropGuard AI | Crop Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

MODEL_PATH = Path("crop_disease_model.keras")
LABELS_PATH = Path("class_names.npy")


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(
        LABELS_PATH,
        allow_pickle=True
    )
    return model, class_names


# --------------------------------------------------
# DISEASE-SPECIFIC RECOMMENDATIONS
# --------------------------------------------------

def get_recommendations(disease):

    name = str(disease).lower().replace("_", " ")
    name = name.replace("-", " ")

    # Healthy plant
    if "healthy" in name:
        return (
            "Healthy",
            [
                "Continue monitoring the crop regularly.",
                "Maintain proper irrigation and drainage.",
                "Use balanced fertilizers according to crop needs.",
                "Inspect leaves regularly for new symptoms."
            ]
        )

    # Leaf curl
    if "leaf curl" in name or "leafcurl" in name:
        return (
            "Possible leaf curl disease",
            [
                "Inspect the undersides of leaves for whiteflies.",
                "Check nearby plants for curling and yellowing symptoms.",
                "Use locally recommended methods to manage insect vectors.",
                "Ask an agricultural expert to confirm the disease "
                "and suitable management."
            ]
        )

    # Early blight
    if "early blight" in name:
        return (
            "Possible early blight",
            [
                "Remove infected plant debris where practical.",
                "Avoid splashing irrigation water onto leaves.",
                "Maintain suitable spacing to improve airflow.",
                "Ask an agricultural expert about locally approved treatments."
            ]
        )

    # Late blight
    if "late blight" in name:
        return (
            "Possible late blight",
            [
                "Inspect nearby plants for spreading leaf lesions.",
                "Avoid prolonged leaf wetness where practical.",
                "Manage affected plant material according to local guidance.",
                "Seek prompt agricultural advice about appropriate control."
            ]
        )

    # Bacterial spot
    if "bacterial spot" in name:
        return (
            "Possible bacterial spot",
            [
                "Avoid handling plants when leaves are wet.",
                "Clean tools between plants.",
                "Avoid splashing water from affected plants onto healthy plants.",
                "Consult an agricultural expert about suitable management."
            ]
        )

    # Powdery mildew
    if "powdery mildew" in name:
        return (
            "Possible powdery mildew",
            [
                "Improve airflow around the plants.",
                "Avoid excessive nitrogen fertilizer.",
                "Monitor new leaves for powdery growth.",
                "Ask an agricultural expert about suitable local treatment."
            ]
        )

    # Rust
    if "rust" in name:
        return (
            "Possible rust disease",
            [
                "Check nearby leaves for rust-colored spots or pustules.",
                "Remove heavily affected leaves when appropriate.",
                "Improve airflow and avoid prolonged leaf wetness.",
                "Seek locally suitable disease-management advice."
            ]
        )

    # Mosaic virus
    if "mosaic" in name:
        return (
            "Possible mosaic virus symptoms",
            [
                "Inspect plants for mottled leaves and distorted growth.",
                "Check for insect vectors such as aphids or whiteflies.",
                "Clean tools and avoid transferring plant sap between plants.",
                "Ask an agricultural expert to confirm the diagnosis."
            ]
        )

    # Leaf mold
    if "leaf mold" in name:
        return (
            "Possible leaf mold",
            [
                "Improve ventilation around the plants.",
                "Avoid prolonged moisture on leaves.",
                "Monitor the undersides of leaves for further symptoms.",
                "Consult an agricultural expert about suitable management."
            ]
        )

    # Septoria
    if "septoria" in name:
        return (
            "Possible Septoria leaf spot",
            [
                "Remove infected plant debris where practical.",
                "Avoid overhead watering that splashes soil onto leaves.",
                "Maintain suitable spacing between plants.",
                "Seek local agricultural advice if symptoms spread."
            ]
        )

    # Target spot
    if "target spot" in name:
        return (
            "Possible target spot",
            [
                "Inspect nearby leaves for expanding circular lesions.",
                "Improve airflow and avoid prolonged leaf wetness.",
                "Remove affected plant debris where appropriate.",
                "Consult an agricultural expert for confirmation."
            ]
        )

    # Anthracnose
    if "anthracnose" in name:
        return (
            "Possible anthracnose",
            [
                "Remove affected plant debris where practical.",
                "Avoid splashing water between plants.",
                "Use clean planting material and tools.",
                "Ask an agricultural expert about crop-specific management."
            ]
        )

    # General fallback for diseases not yet mapped
    return (
        "Disease category needs confirmation",
        [
            f"Model-predicted category: {disease}.",
            "Compare the symptoms with a trusted crop-disease guide.",
            "Inspect nearby plants for similar symptoms.",
            "Consult a local agricultural expert before applying treatments."
        ]
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "score" not in st.session_state:
    st.session_state.score = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌱 CropGuard AI")
st.subheader("AI-Powered Crop Disease Detection and Early Warning System")

st.write(
    "Upload a leaf photo or take a picture to get an experimental "
    "disease prediction and relevant prevention suggestions."
)

st.divider()


# --------------------------------------------------
# SIDEBAR: MANUAL ENVIRONMENTAL CONDITIONS
# --------------------------------------------------

st.sidebar.header("🌦️ Environmental Conditions")

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=0,
    max_value=50,
    value=25
)

humidity = st.sidebar.slider(
    "Relative humidity (%)",
    min_value=0,
    max_value=100,
    value=60
)

st.sidebar.caption(
    "Enter local readings manually. These values are used only "
    "for a demonstration warning, not a validated outbreak forecast."
)


# --------------------------------------------------
# IMAGE UPLOAD AND CAMERA
# --------------------------------------------------

st.header("📷 Upload Leaf Photo or Take Picture")

upload_col, camera_col = st.columns(2)

with upload_col:
    uploaded_file = st.file_uploader(
        "Choose or drag and drop a leaf photo",
        type=["jpg", "jpeg", "png", "webp"]
    )

with camera_col:
    camera_photo = st.camera_input(
        "Use Camera to Capture a Leaf"
    )

photo = uploaded_file if uploaded_file is not None else camera_photo


# --------------------------------------------------
# IMAGE PREVIEW AND DISEASE PREDICTION
# --------------------------------------------------

if photo is not None:

    st.image(
        photo,
        caption="Selected leaf photo",
        width=350
    )

    if st.button(
        "🔍 Detect Disease",
        use_container_width=True
    ):

        st.session_state.prediction = None
        st.session_state.score = None
        st.session_state.audio_bytes = None

        try:

            if not MODEL_PATH.exists():
                st.error(
                    "Model file is missing: "
                    "model/crop_disease_model.keras"
                )

            elif not LABELS_PATH.exists():
                st.error(
                    "Class labels are missing: "
                    "model/class_names.npy"
                )

            else:

                with st.spinner("Analyzing the leaf image..."):

                    model, class_names = load_model()

                    image = Image.open(photo).convert("RGB")
                    image = image.resize((128, 128))

                    image_array = np.asarray(
                        image,
                        dtype=np.float32
                    ) / 255.0

                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    best_index = int(np.argmax(predictions))

                    prediction = str(class_names[best_index])
                    score = float(predictions[best_index]) * 100

                    st.session_state.prediction = prediction
                    st.session_state.score = score

        except Exception as error:
            st.error(
                f"Could not analyze the image: {error}"
            )


# --------------------------------------------------
# PREDICTION RESULTS
# --------------------------------------------------

if st.session_state.prediction is not None:

    st.divider()
    st.header("📊 Prediction Results")

    result_col, score_col = st.columns(2)

    with result_col:
        st.metric(
            "Predicted Category",
            st.session_state.prediction
        )

    with score_col:
        st.metric(
            "Model Score",
            f"{st.session_state.score:.2f}%"
        )

    st.warning(
        "This is an experimental prediction, not a confirmed diagnosis. "
        "The model can make mistakes, especially on unfamiliar images."
    )

    # --------------------------------------------------
    # PREVENTION RECOMMENDATIONS
    # --------------------------------------------------

    st.divider()
    st.header("🌿 Prevention and Management Recommendations")

    status, advice_list = get_recommendations(
        st.session_state.prediction
    )

    st.write(f"**Assessment:** {status}")

    for advice in advice_list:
        st.markdown(f"- {advice}")


    # --------------------------------------------------
    # ENVIRONMENTAL WARNING
    # --------------------------------------------------

    st.divider()
    st.header("🌦️ Environmental Risk Indicator")

    st.write(f"**Temperature:** {temperature} °C")
    st.write(f"**Relative humidity:** {humidity}%")

    if humidity >= 80 and 20 <= temperature <= 30:

        st.warning(
            "The entered conditions meet this demonstration's "
            "high-humidity rule. Monitor crops carefully. "
            "This does not confirm that a disease is present."
        )

    elif humidity >= 80 or temperature >= 35:

        st.info(
            "One entered environmental value meets a demonstration "
            "alert condition. Check actual crop conditions and "
            "local agricultural guidance."
        )

    else:

        st.success(
            "The demonstration rule did not flag these readings. "
            "This does not guarantee that the crop is disease-free."
        )


    # --------------------------------------------------
    # TELUGU AUDIO ASSISTANT
    # --------------------------------------------------

    st.divider()
    st.header("🔊 Telugu Audio Assistant")

    st.write(
        "Generate a Telugu audio summary of the model's prediction."
    )

    if st.button(
        "▶️ Generate Telugu Audio",
        use_container_width=True
    ):

        try:

            with st.spinner("Generating Telugu speech..."):

                result_text = st.session_state.prediction

                telugu_text = (
                    f"మీ పంట ఆకు చిత్రాన్ని విశ్లేషించాము. "
                    f"మోడల్ అంచనా వర్గం: {result_text}. "
                    f"ఈ ఫలితం ఒక అంచనా మాత్రమే. "
                    f"ఖచ్చితమైన నిర్ధారణ కోసం వ్యవసాయ నిపుణులను సంప్రదించండి."
                )

                audio_buffer = io.BytesIO()

                speech = gTTS(
                    text=telugu_text,
                    lang="te"
                )

                speech.write_to_fp(audio_buffer)
                audio_buffer.seek(0)

                st.session_state.audio_bytes = (
                    audio_buffer.getvalue()
                )

        except Exception as error:

            st.error(
                "Could not generate Telugu audio. "
                "Check your internet connection. "
                f"Details: {error}"
            )

    if st.session_state.audio_bytes is not None:

        st.audio(
            st.session_state.audio_bytes,
            format="audio/mp3"
        )

else:

    st.info(
        "Upload a leaf photo or use your camera, "
        "then click Detect Disease."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CropGuard AI | Student demonstration project. "
    "Predictions and environmental rules are experimental; "
    "confirm crop diseases with a qualified agricultural expert."
)
