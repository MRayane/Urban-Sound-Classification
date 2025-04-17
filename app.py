import gradio as gr
import librosa as lb
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("Model/Sound_classification_model.h5")

# Your class labels — update this based on your training
classes = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark', 'Drilling',
           'Engine Idling', 'Gun Shot', 'Jackhammer', 'Siren', 'Street Music']

# Prediction function
def predict_sound(file):
    # Load audio
    data, sr = lb.load(file, sr=None)  # sr=None to preserve original
    mfccs = lb.feature.mfcc(y=data, sr=sr, n_mfcc=128)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_mean = mfccs_mean.reshape(1, -1)
    
    # Prediction
    predictions = model.predict(mfccs_mean)
    predicted_class = np.argmax(predictions)
    pred_prob = predictions[0][predicted_class]

    return f"{classes[predicted_class]} ({pred_prob * 100:.2f}%)"

# Gradio interface
interface = gr.Interface(
    fn=predict_sound,
    inputs=gr.Audio(type="filepath"),  # This gives you a local path to the file
    outputs="text",
    title="Urban Sound Classifier",
    description="Upload an audio file and get the predicted class :"
)

# Launch the app
interface.launch()
