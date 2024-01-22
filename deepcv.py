import cv2
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow import keras


import tensorflow as tf
from tensorflow.keras.models import load_model





# Load the Detectron2 model
config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"  # Update with the correct config file
model_weights = "C:\\Users\\LENOVO\\Downloads\\model_final.pth"  # Update with the path to your trained model weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set the confidence threshold for predictions
cfg.MODEL.WEIGHTS = model_weights
predictor = DefaultPredictor(cfg)



# Load the saved model
severity_model_path = 'C:\\Users\\LENOVO\\Downloads\\damage_classification_model.h5'
severity_model = load_model(severity_model_path)

def preprocess_for_severity(uploaded_image):
    try:
        # Convert PIL Image to NumPy array
        img_array = np.array(uploaded_image)

        # Resize the image to match the model input size
        img_array = tf.image.resize(img_array, (150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img_array)

        # Normalize pixel values to be between 0 and 1
        img_array = img_array / 255.0

        # Expand dimensions to match the model's expected input shape
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None


# Function to preprocess the uploaded image
def preprocess_uploaded_image(uploaded_image):
    try:
        # Convert PIL Image to NumPy array
        img_array = np.array(uploaded_image)

        # Resize the image to match the model input size
        img_array = cv2.resize(img_array, (150, 150))

        # Normalize pixel values to be between 0 and 1
        img_array = img_array / 255.0

        # Expand dimensions to match the model's expected input shape
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Function to load and predict using the model
def classify_AI(uploaded_image, AI_classifier_path):
    # Preprocess the uploaded image
    preprocessed_img = preprocess_uploaded_image(uploaded_image)

    if preprocessed_img is None:
        return None

    # Load the saved model
    AI_classifier = keras.models.load_model(AI_classifier_path)

    # Make predictions
    predictions = AI_classifier.predict(preprocessed_img)

    return predictions

# Streamlit code
st.markdown('## AI generated image classifier')
uploaded_file = st.file_uploader("Choose a picture", type=["jpg", "png", "jpeg", "gif"])

if uploaded_file is not None:
    # Check if the uploaded file is an image
    if uploaded_file.type.startswith("image"):
        # Display the uploaded image using Streamlit
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Make predictions using the loaded model
        AI_classifier_path = 'C:\\Users\\LENOVO\\Downloads\\AI_Or_Not.h5'  # Update with your saved model path
        predictions = classify_AI(uploaded_image, AI_classifier_path)

        if predictions is not None:
            # Assuming binary classification, you may round the prediction to get the final class
            rounded_predictions = np.round(predictions)
            st.write("Raw Predictions:", predictions)
            st.write("Rounded Predictions:", rounded_predictions)
            # Assuming binary classification
            if rounded_predictions is not None and len(rounded_predictions) > 0:
                predicted_class = "fake" if rounded_predictions[0][0] == 1 else "real"

                st.write(f"The predicted class is: {predicted_class}")

            
        #car severity damage    
        preprocessed_img = preprocess_uploaded_image(uploaded_image)
        if preprocessed_img is not None:
        # Make predictions using the loaded model
            predictions = severity_model.predict(preprocessed_img)
            # Assuming categorical classification, get the predicted class name
            class_names = ["Minor", "Moderate", "Severe"]
            predicted_class = class_names[np.argmax(predictions)]

            st.write(f"The predicted class is: {predicted_class}")
        
        # damage detection 
        image_array = torch.tensor(np.array(uploaded_image))
        outputs = predictor(image_array)
        # Visualize the predictions
        v = Visualizer(image_array[:, :, ::-1], metadata=val_metadata_dicts, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.image(out.get_image()[:, :, ::-1], caption='Detected Damage', use_column_width=True)

            

