import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import base64

def main_page():
    # Set the page configuration
    st.set_page_config(page_title="Fake News Detector", layout="centered")

    # Custom CSS for modern and classy look
    st.markdown("""
        <style>
        header[data-testid="stHeader"] {
            background-color: #3E3232;  /* Set your desired header color here */
        }
        
        /* Change the header text color */
        header[data-testid="stHeader"] .stMarkdown {
            color: white;  /* Set header text color to white */
        }

        /* Ensure the background color covers the entire app */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #3E3232;  /* Light gray background */
        }
                
        .description {
            color: white;
            font-size: 1em;
            margin-left: -120px;
        }
                
        .big-title {
            font-size: 4em;  
            font-weight: bold;
            text-align: center;
            color: white;
            margin-top: -70px;  
        }
        .video-section {
            display: flex;
            justify-content: center;
            margin-top: 20px;

        }

        .stButton button {
            background-color: #A87C7C;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #503C3C;
        }
        hr.custom-line {
            height: 1px; /* Adjust the height */
            background: white; /* Change color */
            width: 117.5%; /* Adjust the width */
            margin-left: -120px;
        }
        </style>
        """, unsafe_allow_html=True)

   # Title Section
    st.markdown('<div class="big-title">Fake News Detector</div>', unsafe_allow_html=True)

    # Horizontal Line
    st.markdown('<hr class="custom-line">', unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
    # Left side: Brief description
        st.markdown('<div class="description">Hoax News Detector adalah sebuah website yang dirancang untuk membantu pengguna memverifikasi kebenaran berita atau informasi yang tersebar di internet. Melalui platform ini, pengguna dapat memasukkan tautan berita atau potongan informasi yang meragukan untuk mendapatkan analisis otomatis mengenai tingkat keakuratan atau potensi hoaks dari konten tersebut. Website ini menggunakan teknologi kecerdasan buatan dan basis data fakta terbaru untuk memberikan penilaian yang cepat dan akurat, sehingga membantu masyarakat menghindari penyebaran berita palsu serta mendukung penyebaran informasi yang benar dan dapat dipercaya.</div>', unsafe_allow_html=True)

    # Add horizontal line between columns
    st.markdown('<hr class="custom-line">', unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .video-container {
            width: 100%;
            display: flex;
            justify-content: center;
        }
        .video-container video {
            width: 100%; 
            height: auto;
        }
        </style>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video("Demo(Edit).mp4")
        st.markdown('</div>', unsafe_allow_html=True)


    # Button Section
    st.markdown('<div class="button-section">', unsafe_allow_html=True)
    if st.button('Try it now'):
        st.experimental_set_query_params(page="second")  # Change to "second" to navigate
    st.markdown('</div>', unsafe_allow_html=True)

def second_page():
        # Set the page configuration
    st.set_page_config(page_title="Fake News Detector", layout="centered")

    # Custom CSS for modern and classy look
    st.markdown("""
        <style>
        header[data-testid="stHeader"] {
            background-color: #3E3232;  /* Set your desired header color here */
        }
        
        /* Change the header text color */
        header[data-testid="stHeader"] .stMarkdown {
            color: white;  /* Set header text color to white */
        }

        /* Ensure the background color covers the entire app */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #3E3232;  /* Light gray background */
        }
                
        .big-title {
            font-size: 4em;  
            font-weight: bold;
            text-align: center;
            color: white;
            margin-top: -70px;  
        }
        .button-section {
            display: flex;
            justify-content: center;
            margin-top: 40px;
        }
        .stButton button {
            background-color: #A87C7C;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #503C3C;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # st.markdown('<div class="big-title">Fake News Detector</div>', unsafe_allow_html=True)    

    # Download necessary NLTK data
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load stopwords
    try:
        stop_words = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('indonesian'))

    try:
        lm = WordNetLemmatizer()
    except LookupError:
        nltk.download('wordnet')
        lm = WordNetLemmatizer()

    # Load dataset
    df = pd.read_csv("data.csv", delimiter=';')

    # Drop unnecessary columns and handle missing values
    df = df.dropna()
    df.reset_index(inplace=True)
    df = df.drop(['Tanggal', 'Narasi'], axis=1)

    # Initialize Lemmatizer
    lm = WordNetLemmatizer()

    # Text Preprocessing
    corpus = []
    for i in range(len(df)):
        review = re.sub('[^a-zA-Z0-9]', ' ', df['Judul'][i])  # Corrected regex
        review = review.lower()
        review = review.split()
        review = [lm.lemmatize(x) for x in review if x not in stop_words]
        review = " ".join(review)
        corpus.append(review)

    # TF-IDF Vectorization
    tf = TfidfVectorizer()
    x = tf.fit_transform(corpus).toarray()
    y = df['Label']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10, stratify=y)

    # Model training
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    rf.fit(x_train, y_train)

    # Evaluate model
    class Evaluation:
        
        def __init__(self, model, x_train, x_test, y_train, y_test):
            self.model = model
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test
            
        def train_evaluation(self):
            y_pred_train = self.model.predict(self.x_train)
            
            acc_scr_train = accuracy_score(self.y_train, y_pred_train)
            print("Accuracy Score On Training Data Set:", acc_scr_train)
            print()
            
            con_mat_train = confusion_matrix(self.y_train, y_pred_train)
            print("Confusion Matrix On Training Data Set:\n", con_mat_train)
            print()
            
            class_rep_train = classification_report(self.y_train, y_pred_train)
            print("Classification Report On Training Data Set:\n", class_rep_train)
            
        def test_evaluation(self):
            y_pred_test = self.model.predict(self.x_test)
            
            acc_scr_test = accuracy_score(self.y_test, y_pred_test)
            print("Accuracy Score On Testing Data Set:", acc_scr_test)
            print()
            
            con_mat_test = confusion_matrix(self.y_test, y_pred_test)
            print("Confusion Matrix On Testing Data Set:\n", con_mat_test)
            print()
            
            class_rep_test = classification_report(self.y_test, y_pred_test)
            print("Classification Report On Testing Data Set:\n", class_rep_test)

    Evaluation(rf, x_train, x_test, y_train, y_test).train_evaluation()
    Evaluation(rf, x_train, x_test, y_train, y_test).test_evaluation()

    # Preprocessing class for user input
    class Preprocessing:
        
        def __init__(self, data):
            self.data = data
            
        def text_preprocessing_user(self):
            lm = WordNetLemmatizer()
            pred_data = [self.data]    
            preprocess_data = []
            for data in pred_data:
                review = re.sub('[^a-zA-Z0-9]', ' ', data)  # Corrected regex
                review = review.lower()
                review = review.split()
                review = [lm.lemmatize(x) for x in review if x not in stop_words]
                review = " ".join(review)
                preprocess_data.append(review)
            return preprocess_data   

    # Streamlit App
    st.title('Fake News Detector')
    input_text = st.text_input('Enter news Article')

    def prediction(input_text):
        # Preprocess and transform the input text
        input_data = Preprocessing(input_text).text_preprocessing_user()
        input_data = tf.transform(input_data)
        
        # Predict using the trained model
        prediction = rf.predict(input_data)
        return prediction[0]

    if input_text:
        pred = prediction(input_text)
        if pred == 1:
            st.title('The News is Fake')
        else:
            st.title('The News Is Real')

# Button to navigate back to the main page
    if st.button("Back to Main Page"):
        st.experimental_set_query_params(page="main")


# Handle redirection based on query parameters
query_params = st.experimental_get_query_params()

# Check the 'page' query parameter to determine which page to show
if query_params.get("page") == ["second"]:
    second_page()  # Show the second page
else:
    main_page()  # Show the main page by default
