import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import speech_recognition as sr
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Helper Functions

# 1. Voice Input with SpeechRecognition
def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Please speak.")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError:
            st.error("Speech recognition service error.")
    return ""

# 2. Load and Classify Pill Image with ResNet Model
@st.cache_resource
def load_pill_model():
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 86)  # Replace NUM_CLASSES with your class count
    model.load_state_dict(torch.load("pill_identifier.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def classify_pill_image(img, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()  # Map to class label externally

# 3. Text-based Drug Recommendation with Cosine Similarity
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    text = text.lower()
    return text

def recommend_drugs_by_symptom(user_input, df, top_n=5):
    vectorizer = TfidfVectorizer()
    condition_vectors = vectorizer.fit_transform(df['condition'])
    user_vec = vectorizer.transform([clean_text(user_input)])
    similarity = cosine_similarity(user_vec, condition_vectors)
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['drugName', 'condition', 'review', 'rating']]

# Load dataset
@st.cache_resource
def load_data():
    df = pd.read_csv('train.csv')
    df = df[['drugName', 'condition', 'review', 'rating', 'usefulCount']]
    df.dropna(inplace=True)
    return df

df = load_data()

# Streamlit Layout

# Sidebar
st.sidebar.title("Drug Recommendation Chatbot")
user_input = st.sidebar.text_input("Enter symptom or condition (e.g., headache, insomnia)")

# Initialize session state variables
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_responses' not in st.session_state:
    st.session_state.user_responses = {}

# Chatbot Logic
def chatbot_interaction():
    if st.session_state.step == 0:
        st.write("👋 Hello! I'm here to help you find the right drug.")
        st.session_state.step += 1
        st.session_state.user_responses['symptom'] = st.text_input("Please describe your symptom or condition:")

    elif st.session_state.step == 1 and 'symptom' in st.session_state.user_responses:
        st.write("🔍 Thanks for your input! Let's narrow down the options.")
        st.session_state.step += 1
        st.session_state.user_responses['severity'] = st.selectbox("On a scale of 1 to 5, how severe is the condition?", [1, 2, 3, 4, 5])

    elif st.session_state.step == 2 and 'severity' in st.session_state.user_responses:
        st.write("👌 Great! One last question.")
        st.session_state.step += 1
        st.session_state.user_responses['duration'] = st.selectbox("How long have you been experiencing this?", ['Less than a week', '1-2 weeks', 'More than 2 weeks'])

    elif st.session_state.step == 3 and 'duration' in st.session_state.user_responses:
        st.write("🔎 Searching for the best drugs based on your inputs...")
        symptom = st.session_state.user_responses['symptom']
        recommendations = recommend_drugs_by_symptom(symptom, df)
        st.write("### Recommended Drugs:")
        st.write(recommendations)

        # Feedback mechanism
        st.write("📝 Please rate the recommendations on a scale of 1 to 5 (1 = not helpful, 5 = very helpful):")
        rating = st.slider("Rating", 1, 5)
        st.session_state.user_responses['rating'] = rating
        if st.button("Submit Feedback"):
            st.write(f"Thank you for your feedback! You rated the recommendations {rating}/5.")

    else:
        st.write("🔄 You can go back and edit your responses.")
        st.button("Restart Chat", on_click=lambda: reset_chat())

# Reset chat logic
def reset_chat():
    st.session_state.step = 0
    st.session_state.user_responses = {}

# Main layout and interaction flow
st.title("Drug Recommendation System 🩺💊")
chatbot_interaction()

# Pill Image Upload and Classification
st.sidebar.header("Pill Image Identification")
uploaded_file = st.sidebar.file_uploader("Upload a pill image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Pill Image", use_column_width=True)

    model = load_pill_model()
    pill_class = classify_pill_image(image, model)
    st.success(f"🔎 Pill identified as class {pill_class}")

# Voice Input (Optional)
st.sidebar.header("Voice Input")
if st.sidebar.button("Start Listening (Speak Now)"):
    user_input = transcribe_audio()
    if user_input:
        recommendations = recommend_drugs_by_symptom(user_input, df)
        st.write(f"🔍 Searching for drugs for **{user_input}**...")
        st.write(recommendations)

