import { useRef, useState, useEffect } from 'react';

function Webcam({ onEmotionChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState('neutral');

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240 }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setWebcamActive(true);
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      alert("Could not access webcam. Please check permissions.");
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setWebcamActive(false);
    }
  };

  // Capture frame and send for emotion analysis
  const captureFrame = () => {
    if (videoRef.current && canvasRef.current && webcamActive) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const imageData = canvas.toDataURL('image/jpeg');
      return imageData.split(',')[1]; // Remove the data URL prefix
    }
    return null;
  };

  // Emotion detection loop using useEffect to ensure interval persists
  useEffect(() => {
    let detectEmotionInterval;

    if (webcamActive) {
      detectEmotionInterval = setInterval(async () => {
        const frameData = captureFrame();
        if (frameData) {
          try {
            const response = await fetch('http://localhost:8000/analyze-emotion', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ image: frameData }),
            });

            if (response.ok) {
              const data = await response.json();
              setCurrentEmotion(data.emotion);
              if (onEmotionChange) {
                onEmotionChange(data.emotion); // Pass emotion to parent
              }
            }
          } catch (error) {
            console.error('Error analyzing emotion:', error);
          }
        }
      }, 1000);
    }

    return () => clearInterval(detectEmotionInterval); // Cleanup on unmount
  }, [webcamActive]); // Re-run when `webcamActive` changes

  useEffect(() => {
    startWebcam();
    return () => stopWebcam();
  }, []);

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline className='border rounded-md shadow-md' />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <p className='mt-2 font-semibold'>Detected Emotion: {currentEmotion}</p>
    </div>
  );
}

export default Webcam;
