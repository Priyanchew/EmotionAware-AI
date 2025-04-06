# EmotionAware-AI

EmotionAware-AI is an open-source project designed to detect and interpret human emotions through facial expressions and speech analysis. By leveraging advanced machine learning techniques, this system aims to enhance human-computer interactions by providing emotionally intelligent responses.

## Features

- **Facial Emotion Recognition:** Analyzes facial expressions to identify emotions such as happiness, sadness, anger, surprise, and more.
- **Speech Emotion Recognition:** Processes vocal inputs to detect emotional states conveyed through tone and pitch.
- **Real-time Processing:** Capable of analyzing and responding to emotional cues in real-time.
- **Modular Architecture:** Organized into distinct components for facial analysis (`face`), speech analysis (`speech`), backend processing (`backend`), and frontend interface (`frontend`).

## Installation

To set up the EmotionAware-AI project locally, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Priyanchew/EmotionAware-AI.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd EmotionAware-AI
   ```

3. **Install Dependencies:**

   - **Backend and Speech Analysis:**

     Navigate to the respective directories and install the required Python packages:

     ```bash
     cd backend
     pip install -r requirements.txt
     ```

     ```bash
     cd ../speech
     pip install -r requirements.txt
     ```

   - **Frontend:**

     Navigate to the frontend directory and install the necessary Node.js packages:

     ```bash
     cd ../frontend
     npm install
     ```

4. **Run the Application:**

   - **Backend:**

     ```bash
     cd ../backend
     python app.py
     ```

   - **Frontend:**

     In a new terminal window:

     ```bash
     cd frontend
     npm start
     ```

## Usage

Once the application is running:

1. **Access the Frontend Interface:**

   Open your web browser and navigate to `http://localhost:3000` to interact with the EmotionAware-AI system.

2. **Emotion Detection:**

   - **Facial Analysis:** Use your device's camera to allow the system to analyze your facial expressions.
   - **Speech Analysis:** Provide vocal input through your device's microphone for speech emotion recognition.

3. **Real-time Feedback:**

   The system will display the detected emotions in real-time, providing insights into the emotional state conveyed through facial expressions and speech.

## Contributing

Contributions to EmotionAware-AI are welcome. To contribute:

1. **Fork the Repository:**

   Click on the 'Fork' button at the top right corner of the repository page.

2. **Create a New Branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes:**

   Implement your feature or fix and commit the changes with a descriptive message.

4. **Push to Your Fork:**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Submit a Pull Request:**

   Navigate to the original repository and click on 'New Pull Request' to submit your changes for review.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the contributors and the open-source community for their invaluable support and resources.

---

*Note: For optimal performance, ensure that your device has a functional camera and microphone. The application may require permissions to access these devices.*

--- 
