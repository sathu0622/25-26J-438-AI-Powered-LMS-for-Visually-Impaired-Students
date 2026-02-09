import { useState, useCallback, useRef } from 'react';

/**
 * Mock Speech Recognition Hook
 * Simulates voice input without requiring microphone permissions
 * Perfect for demos and testing voice-first interfaces
 * 
 * Usage: Click the microphone button to trigger a text input modal
 * where users can type what they would say
 */
export const useMockSpeechRecognition = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [isSupported] = useState(true); // Always supported in mock mode
  const [error, setError] = useState<string | null>(null);
  const [permissionDenied] = useState(false); // Never denied in mock mode
  const mockInputRef = useRef<string>('');

  const startListening = useCallback(() => {
    setIsListening(true);
    setTranscript('');
    setInterimTranscript('');
    setError(null);
    mockInputRef.current = '';

    console.log('[Mock STT] Listening started (waiting for text input)...');

    // Show custom prompt for text input
    // In a real implementation, this would be handled by a modal component
    const promptText = 'DEMO MODE: Enter what you would say:';
    
    // Use a timeout to simulate async behavior
    setTimeout(() => {
      const userInput = prompt(promptText);
      
      if (userInput) {
        mockInputRef.current = userInput;
        
        // Simulate interim results
        setInterimTranscript(userInput);
        
        // After a brief delay, finalize the transcript
        setTimeout(() => {
          setTranscript(userInput);
          setInterimTranscript('');
          setIsListening(false);
          console.log(`[Mock STT] Recognized: "${userInput}"`);
        }, 500);
      } else {
        // User cancelled
        setIsListening(false);
        console.log('[Mock STT] Cancelled by user');
      }
    }, 100);
  }, []);

  const stopListening = useCallback(() => {
    setIsListening(false);
    setInterimTranscript('');
    console.log('[Mock STT] Listening stopped');
  }, []);

  const resetTranscript = useCallback(() => {
    setTranscript('');
    setInterimTranscript('');
    mockInputRef.current = '';
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Simulate different voice commands for testing
  const simulateCommand = useCallback((command: string) => {
    setTranscript(command);
    console.log(`[Mock STT] Simulated command: "${command}"`);
  }, []);

  return {
    isListening,
    transcript,
    interimTranscript,
    startListening,
    stopListening,
    resetTranscript,
    isSupported,
    error,
    permissionDenied,
    clearError,
    simulateCommand, // Extra method for testing
  };
};
