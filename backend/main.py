from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import io
from PIL import Image
import google.generativeai as genai
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import dotenv

dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Load emotion recognition model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

# Emotion mapping
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

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
        raise HTTPException(status_code=500, detail=f"Error analyzing emotion: {str(e)}")

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = get_llm_response(request.message, request.emotion)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting response: {str(e)}")

# Emotion detection function
def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return "face not detected"
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
    face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0)
    prediction = model.predict(face_roi)
    return emotion_dict[np.argmax(prediction)]

# LLM response function
def get_llm_response(message, emotion):
    prompt = f"""
    You are a helpful and empathetic chatbot. The user is interacting with you through a webcam, and their detected emotion is "{emotion}".
    Based on this emotion and their message: "{message}", provide a relevant and empathetic response.
    """
    try:
        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
        response = model_gemini.generate_content(prompt)
        return response.text if response.text else "Sorry, I couldn't generate a response this time."
    except Exception as e:
        print(f"ERROR in Gemini API call: {e}")
        return "Error generating response."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
