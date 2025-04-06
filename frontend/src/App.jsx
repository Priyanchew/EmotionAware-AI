import { useState } from 'react';
import Chatbot from './components/Chatbot';
import Webcam from './components/Webcam';

function App() {
  const [currentEmotion, setCurrentEmotion] = useState('neutral');

  return (
      <>
    <div className='flex flex-col min-h-full w-full max-w-3xl mx-auto px-4 relative'>
      <header className='sticky top-0 shrink-0 z-20 bg-white'>
        <div className='flex flex-col h-full w-full gap-1 pt-4 pb-2'>
          <h1 className='font-urbanist text-[1.65rem] font-semibold'>Emotion Aware AI</h1>
        </div>
      </header>

      <Chatbot emotion={currentEmotion} />
    </div>
      {/* Position webcam in the bottom-right corner where the box is drawn */}
      <div className='absolute bottom-24 right-4 z-10 w-60 h-36'>
        <Webcam onEmotionChange={setCurrentEmotion} />
      </div>
          </>
  );
}

export default App;