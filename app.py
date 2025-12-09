import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
@st.cache_resource
def load_models():
    try:
        svm_model = joblib.load('models/svm_model.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
        return svm_model, rf_model, scaler, cnn_model
    except FileNotFoundError:
        st.error("Models not found. Please train the models first.")
        return None, None, None, None

def preprocess_image_ml(image, scaler):
    # Resize to 64x64 for ML models
    img = cv2.resize(np.array(image), (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = gray.flatten().reshape(1, -1)
    features_scaled = scaler.transform(features)
    return features_scaled

def preprocess_image_dl(image):
    # Resize to 128x128 for CNN
    img = cv2.resize(np.array(image), (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    st.title("Brain Tumor Detection App")
    st.write("Upload an MRI image to detect brain tumors using Machine Learning and Deep Learning models.")

    # Load models
    svm_model, rf_model, scaler, cnn_model = load_models()
    if svm_model is None:
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess for ML
        features_ml = preprocess_image_ml(image, scaler)

        # Preprocess for DL
        features_dl = preprocess_image_dl(image)

        # Predictions
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("SVM Prediction")
            pred_svm = svm_model.predict(features_ml)[0]
            prob_svm = svm_model.decision_function(features_ml)[0]
            st.write(f"Prediction: {'Tumor' if pred_svm == 1 else 'No Tumor'}")
            st.write(".4f")

        with col2:
            st.subheader("Random Forest Prediction")
            pred_rf = rf_model.predict(features_ml)[0]
            prob_rf = rf_model.predict_proba(features_ml)[0][1]
            st.write(f"Prediction: {'Tumor' if pred_rf == 1 else 'No Tumor'}")
            st.write(".4f")

        with col3:
            st.subheader("CNN Prediction")
            pred_cnn = (cnn_model.predict(features_dl) > 0.5).astype(int)[0][0]
            prob_cnn = cnn_model.predict(features_dl)[0][0]
            st.write(f"Prediction: {'Tumor' if pred_cnn == 1 else 'No Tumor'}")
            st.write(".4f")

        # Ensemble prediction
        st.subheader("Ensemble Prediction")
        ensemble_pred = (pred_svm + pred_rf + pred_cnn) >= 2
        ensemble_prob = (prob_svm + prob_rf + prob_cnn) / 3
        st.write(f"Ensemble Prediction: {'Tumor' if ensemble_pred else 'No Tumor'}")
        st.write(".4f")

        # Visualization
        st.subheader("Model Comparison")
        models = ['SVM', 'Random Forest', 'CNN']
        predictions = [pred_svm, pred_rf, pred_cnn]
        probabilities = [prob_svm, prob_rf, prob_cnn]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.bar(models, predictions, color=['red' if p == 1 else 'green' for p in predictions])
        ax1.set_title('Predictions (1=Tumor, 0=No Tumor)')
        ax1.set_ylabel('Prediction')

        ax2.bar(models, probabilities, color='blue')
        ax2.set_title('Prediction Probabilities')
        ax2.set_ylabel('Probability')

        st.pyplot(fig)

if __name__ == "__main__":
    main()
