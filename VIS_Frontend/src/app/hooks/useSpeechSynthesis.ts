import { useState, useEffect, useRef, useCallback } from 'react';

export const useSpeechSynthesis = () => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const [voicesLoaded, setVoicesLoaded] = useState(false);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    const supported = 'speechSynthesis' in window;
    setIsSupported(supported);

    if (supported) {
      // Load voices
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {
          setVoicesLoaded(true);
        }
      };

      // Chrome needs this event
      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = loadVoices;
      }

      // Try loading immediately (for other browsers)
      loadVoices();

      // Fallback: mark as loaded after 1 second even if no voices
      const timeout = setTimeout(() => {
        setVoicesLoaded(true);
      }, 1000);

      return () => {
        clearTimeout(timeout);
      };
    }
  }, []);

  const speak = useCallback((text: string, onEnd?: () => void) => {
    if (!isSupported) {
      console.warn('Speech synthesis not supported in this browser');
      onEnd?.();
      return;
    }

    // Wait for voices to load
    if (!voicesLoaded) {
      console.warn('Speech synthesis voices not loaded yet');
      // Retry after a short delay
      setTimeout(() => speak(text, onEnd), 500);
      return;
    }

    try {
      // Cancel any ongoing speech
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      
      // Get available voices and prefer English
      const voices = window.speechSynthesis.getVoices();
      const englishVoice = voices.find(voice => voice.lang.startsWith('en'));
      if (englishVoice) {
        utterance.voice = englishVoice;
      }

      utterance.rate = 0.9; // Slightly slower for better comprehension
      utterance.pitch = 1;
      utterance.volume = 1;
      utterance.lang = 'en-US';

      utterance.onstart = () => {
        setIsSpeaking(true);
        setIsPaused(false);
      };

      utterance.onend = () => {
        setIsSpeaking(false);
        setIsPaused(false);
        onEnd?.();
      };

      utterance.onerror = (event) => {
        // Suppress "interrupted" errors as they're expected when canceling
        if (event.error !== 'interrupted' && event.error !== 'canceled') {
          console.error('Speech synthesis error:', event.error);
        }
        setIsSpeaking(false);
        setIsPaused(false);
        onEnd?.();
      };

      utteranceRef.current = utterance;
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      console.error('Speech synthesis error:', error);
      setIsSpeaking(false);
      setIsPaused(false);
      onEnd?.();
    }
  }, [isSupported, voicesLoaded]);

  const pause = useCallback(() => {
    if (isSupported && isSpeaking) {
      try {
        window.speechSynthesis.pause();
        setIsPaused(true);
      } catch (error) {
        console.error('Error pausing speech:', error);
      }
    }
  }, [isSupported, isSpeaking]);

  const resume = useCallback(() => {
    if (isSupported && isPaused) {
      try {
        window.speechSynthesis.resume();
        setIsPaused(false);
      } catch (error) {
        console.error('Error resuming speech:', error);
      }
    }
  }, [isSupported, isPaused]);

  const stop = useCallback(() => {
    if (isSupported) {
      try {
        window.speechSynthesis.cancel();
        setIsSpeaking(false);
        setIsPaused(false);
      } catch (error) {
        console.error('Error stopping speech:', error);
      }
    }
  }, [isSupported]);

  return {
    speak,
    pause,
    resume,
    stop,
    isSpeaking,
    isPaused,
    isSupported,
    voicesLoaded,
  };
};