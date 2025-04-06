import React, { useState } from 'react';
import { useImmer } from 'use-immer';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';

function Chatbot( { emotion } ) {
  const [messages, setMessages] = useImmer([]);
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!newMessage.trim()) return;
    
    const userMessage = { role: 'user', content: newMessage };
    setMessages(prev => [...prev, userMessage]);
    setNewMessage('');
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message: userMessage.content,
          emotion: emotion}),
      });
      
      if (response.ok) {
        const data = await response.json();
        setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      } else {
        throw new Error('Failed to get response');
      }
    } catch (err) {
      console.error('Error:', err);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, there was an error processing your request.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className='relative grow flex flex-col gap-6 pt-6'>
      {messages.length === 0 && (
        <div className='mt-3 font-urbanist text-primary-blue text-xl font-light space-y-2'>
          <p>👋 Welcome!</p>
          <p>I am powered by gemini and am aware of your emotions.</p>
          <p>Ask me anything!</p>
        </div>
      )}
      <ChatMessages messages={messages} isLoading={isLoading} />
      <ChatInput newMessage={newMessage} isLoading={isLoading} setNewMessage={setNewMessage} submitNewMessage={sendMessage} />
    </div>
  );
}

export default Chatbot;