import { useState, useEffect, useRef, useCallback } from 'react';

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionResult {
  [index: number]: SpeechRecognitionAlternative;
  length: number;
  isFinal: boolean;
}

interface SpeechRecognitionResultList {
  [index: number]: SpeechRecognitionResult;
  length: number;
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognitionInstance;
    webkitSpeechRecognition: new () => SpeechRecognitionInstance;
  }
}

export const useSpeechRecognition = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [isSupported, setIsSupported] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    setIsSupported(!!SpeechRecognition);

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.addEventListener('result', (event: Event) => {
        const e = event as SpeechRecognitionEvent;
        let interim = '';
        let final = '';

        for (let i = e.resultIndex; i < e.results.length; i++) {
          const result = e.results[i];
          const transcriptText = result[0].transcript;

          if (result.isFinal) {
            final += transcriptText + ' ';
          } else {
            interim += transcriptText;
          }
        }

        setInterimTranscript(interim);
        if (final) {
          setTranscript((prev) => prev + final);
        }
      });

      recognition.addEventListener('end', () => {
        setIsListening(false);
      });

      recognition.addEventListener('error', (event: Event) => {
        const e = event as SpeechRecognitionErrorEvent;
        console.error('Speech recognition error:', e.error);
        
        // Handle specific error types
        if (e.error === 'not-allowed' || e.error === 'permission-denied') {
          setPermissionDenied(true);
          setError('Microphone permission denied. Please allow microphone access in your browser settings.');
        } else if (e.error === 'no-speech') {
          setError('No speech detected. Please try again.');
        } else if (e.error === 'audio-capture') {
          setError('No microphone detected. Please connect a microphone.');
        } else if (e.error === 'network') {
          setError('Network error. Speech recognition requires an internet connection.');
        } else {
          setError(`Speech recognition error: ${e.error}`);
        }
        
        setIsListening(false);
      });

      recognitionRef.current = recognition;
    }

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch (error) {
          // Ignore errors during cleanup
        }
      }
    };
  }, []);

  const startListening = useCallback(() => {
    if (!isSupported) {
      setError('Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.');
      return;
    }

    if (permissionDenied) {
      setError('Microphone permission was denied. Please enable microphone access in your browser settings and reload the page.');
      return;
    }

    if (recognitionRef.current && !isListening) {
      try {
        setTranscript('');
        setInterimTranscript('');
        setError(null);
        recognitionRef.current.start();
        setIsListening(true);
      } catch (err) {
        console.error('Error starting speech recognition:', err);
        setError('Unable to start microphone. Please check your browser permissions.');
        setIsListening(false);
      }
    }
  }, [isListening, isSupported, permissionDenied]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      try {
        recognitionRef.current.stop();
        setIsListening(false);
        setInterimTranscript('');
      } catch (err) {
        console.error('Error stopping speech recognition:', err);
      }
    }
  }, [isListening]);

  const resetTranscript = useCallback(() => {
    setTranscript('');
    setInterimTranscript('');
  }, []);

  const clearError = useCallback(() => {
    setError(null);
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
  };
};