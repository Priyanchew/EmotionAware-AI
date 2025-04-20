from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import io
from PIL import Image
import google.generativeai as genai
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import os
import dotenv
from transformers import pipeline, logging as transformers_logging

dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
transformers_logging.set_verbosity_error()

# Configure Google Gemini API
genai.configure(api_key=gemini_api_key)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load emotion recognition model using logic from willitworktho.py
def load_emotion_model(model_json_path="model/model.json", model_weights_path="model/model.weights.h5"):
    print(f"DEBUG: Attempting to load model from {model_json_path} and {model_weights_path}")
    try:
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        print("DEBUG: Model structure loaded from JSON.")
        model.load_weights(model_weights_path)
        print("DEBUG: Model weights loaded.")
        print("Emotion model loaded successfully from json and weights.")
        return model
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        # Depending on the application, you might want to raise the exception
        # or handle it gracefully (e.g., return None and disable emotion detection)
        raise

model = load_emotion_model()

# Load face cascade - Assuming it's in the model directory as well
# Using absolute path from cv2.data if preferred and available
# face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade_path = "model/haarcascade_frontalface_default.xml"
print(f"DEBUG: Attempting to load face cascade from {face_cascade_path}")
if not os.path.exists(face_cascade_path):
    # Fallback to cv2.data.haarcascades if the local one isn't found
    print(f"Warning: Cascade file not found at {face_cascade_path}. Trying default cv2 path.")
    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print(f"ERROR: Failed to load face cascade classifier from {face_cascade_path}. Exiting.")
    # Decide how to handle cascade load failure - exit, or disable detection?
    # For now, let's proceed but detection will fail.
else:
    print("DEBUG: Face cascade loaded successfully.")

# Load Text Sentiment Analysis Model
print("DEBUG: Loading text sentiment analysis model (arpanghoshal/EmoRoBERTa)...")
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="arpanghoshal/EmoRoBERTa")
    print("DEBUG: Text sentiment analysis model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load text sentiment analysis model: {e}")
    sentiment_pipeline = None # Set to None to indicate failure

cv2.ocl.setUseOpenCL(False) # Kept from original

# Emotion mapping - Updated to match willitworktho.py
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Models
class EmotionRequest(BaseModel):
    image: str  # Base64 encoded image

class ChatRequest(BaseModel):
    message: str
    emotion: str = "neutral"  # Default emotion

class ChatResponse(BaseModel):
    response: str

# Emotion analysis endpoint
@app.post("/analyze-emotion")
async def analyze_emotion(request: EmotionRequest):
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        emotion = detect_emotion(cv_image)
        print(f"Detected emotion: {emotion}")
        return {"emotion": emotion}
    except Exception as e:
        print(f"ERROR in /analyze-emotion endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing emotion: {str(e)}")

# Chat endpoint - Modified to include text sentiment analysis
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"DEBUG: Received request for /chat. Message: '{request.message}', Face Emotion: '{request.emotion}'")
    text_sentiment = "unknown" # Default
    if sentiment_pipeline:
        try:
            print(f"DEBUG: Analyzing text sentiment for: '{request.message}'")
            sentiment_result = sentiment_pipeline(request.message)
            # Expected output format: [{'label': '...', 'score': ...}]
            if sentiment_result and isinstance(sentiment_result, list) and 'label' in sentiment_result[0]:
                text_sentiment = sentiment_result[0]['label']
                print(f"DEBUG: Text sentiment analysis result: {text_sentiment}")
            else:
                print(f"DEBUG: Unexpected sentiment analysis output format: {sentiment_result}")
                text_sentiment = "analysis_error"
        except Exception as e:
            print(f"ERROR during text sentiment analysis: {e}")
            text_sentiment = "analysis_error"
    else:
        print("DEBUG: Text sentiment pipeline not available. Skipping analysis.")
        text_sentiment = "model_not_loaded"

    try:
        print(f"DEBUG: Calling get_llm_response with face_emotion='{request.emotion}' and text_sentiment='{text_sentiment}'")
        # Pass both face emotion and text sentiment to LLM
        response = get_llm_response(request.message, request.emotion, text_sentiment)
        print("DEBUG: Received response from LLM.")
        return {"response": response}
    except Exception as e:
        print(f"ERROR in /chat endpoint during LLM call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting response: {str(e)}")

# Emotion detection function - Updated with willitworktho.py logic
def detect_emotion(image):
    print("DEBUG: Entered detect_emotion function.")
    if model is None:
        print("DEBUG: Emotion model is None. Skipping detection.")
        return "model not loaded"
    if face_cascade.empty():
        print("DEBUG: Face cascade is empty. Skipping detection.")
        return "cascade not loaded"

    print("DEBUG: Converting input image to grayscale...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("DEBUG: Detecting faces...")
    # Use the loaded face_cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"DEBUG: Found {len(faces)} faces.")

    if len(faces) == 0:
        print("DEBUG: No faces detected.")
        return "face not detected"

    # Process the largest face found (as in original)
    print("DEBUG: Finding largest face...")
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    print(f"DEBUG: Largest face found at x={x}, y={y}, w={w}, h={h}")

    # Use preprocessing from willitworktho.py
    # Adding a small buffer around the face ROI, similar to willitworktho.py
    print("DEBUG: Extracting face ROI...")
    y_start = max(0, y - 5)
    y_end = min(gray.shape[0], y + h + 5)
    x_start = max(0, x - 5)
    x_end = min(gray.shape[1], x + w + 5)
    roi_gray = gray[y_start:y_end, x_start:x_end]
    print(f"DEBUG: Extracted ROI shape: {roi_gray.shape}")

    # Check if roi_gray is valid before resizing
    if roi_gray.size == 0:
        print("Warning: Empty face ROI detected after extraction.")
        return "face ROI error"

    print("DEBUG: Resizing ROI to (48, 48)...")
    roi_gray = cv2.resize(roi_gray, (48, 48))
    print("DEBUG: Converting ROI to array...")
    roi_gray = img_to_array(roi_gray)
    print("DEBUG: Expanding ROI dimensions...")
    roi_gray = np.expand_dims(roi_gray, axis=0)
    print("DEBUG: Normalizing ROI (dividing by 255.0)...")
    roi_gray = roi_gray.astype('float') / 255.0
    print(f"DEBUG: Final ROI shape for prediction: {roi_gray.shape}")

    try:
        print("DEBUG: Making prediction with model...")
        prediction = model.predict(roi_gray)
        print(f"DEBUG: Prediction raw output: {prediction}")
        max_index = np.argmax(prediction[0])
        print(f"DEBUG: Predicted emotion index: {max_index}")
        detected_emotion = emotion_dict[max_index]
        print(f"DEBUG: Mapped emotion: {detected_emotion}")
        return detected_emotion
    except Exception as e:
        print(f"Error during emotion prediction: {e}")
        return "prediction error"

# LLM response function - Modified to accept text_sentiment
def get_llm_response(message, face_emotion, text_sentiment):
    print(f"DEBUG: Generating LLM prompt with face_emotion='{face_emotion}' and text_sentiment='{text_sentiment}'")
    prompt = f"""
    You are a helpful and empathetic chatbot. The user is interacting with you through a webcam.
    Their detected facial emotion is "{face_emotion}".
    The sentiment analysis of their typed message ("{message}") suggests the text sentiment is "{text_sentiment}".
    Based on both their facial emotion and the text sentiment/content, provide a relevant and empathetic response.
    """
    print("DEBUG: Prompt sent to Gemini")
    try:
        model_gemini = genai.GenerativeModel('gemini-2.5-pro-exp-03-25') # Consider using gemini-pro for potentially better nuance
        response = model_gemini.generate_content(prompt)
        # Check if response has text attribute and content
        response_text = getattr(response, 'text', None)
        print(f"DEBUG: Raw response from Gemini: {response}") # Log the raw response for inspection
        
        if response_text:
            print(f"DEBUG: Extracted text from Gemini response: {response_text}")
            return response_text
        else:
            # Handle cases where the response might be blocked or empty
            print("DEBUG: Gemini response did not contain text. Checking for safety ratings or other issues.")
            # You might want to inspect response.prompt_feedback here
            # feedback = getattr(response, 'prompt_feedback', None)
            # print(f"DEBUG: Gemini prompt feedback: {feedback}")
            return "Sorry, I couldn't generate a suitable response this time."
            
    except Exception as e:
        print(f"ERROR in Gemini API call: {e}")
        return "Sorry, there was an error communicating with the AI assistant."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)