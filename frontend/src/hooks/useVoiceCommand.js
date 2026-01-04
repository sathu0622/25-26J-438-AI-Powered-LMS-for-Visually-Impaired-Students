import { useState, useEffect } from 'react';
import { voiceService } from '../services/voiceService';

export const useVoiceCommand = () => {
  const [isListening, setIsListening] = useState(false);
  const [lastCommand, setLastCommand] = useState('');
  const [transcript, setTranscript] = useState('');

  useEffect(() => {
    // Register transcript listener
    voiceService.onTranscript((recognizedText) => {
      setTranscript(recognizedText);
    });
  }, []);

  const registerCommand = (command, callback) => {
    voiceService.onCommand(command, (transcript) => {
      setLastCommand(transcript);
      callback(transcript);
    });
  };

  const startListening = () => {
    if (voiceService.isSupported()) {
      voiceService.startListening();
      setIsListening(true);
    } else {
      console.warn('Voice recognition not supported');
    }
  };

  const stopListening = () => {
    voiceService.stopListening();
    setIsListening(false);
  };

  const speak = (text) => {
    if (voiceService.isSupported()) {
      voiceService.speak(text);
    }
  };

  return {
    isListening,
    lastCommand,
    transcript,
    startListening,
    stopListening,
    speak,
    registerCommand,
    isSupported: voiceService.isSupported(),
  };
};

export default useVoiceCommand;
