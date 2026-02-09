import { useState, useCallback, useRef } from 'react';

/**
 * Mock Speech Synthesis Hook
 * Simulates text-to-speech without requiring browser speech APIs
 * Perfect for demos and testing accessibility features
 */
export const useMockSpeechSynthesis = () => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isSupported] = useState(true); // Always supported in mock mode
  const [voicesLoaded] = useState(true); // Always loaded in mock mode
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const onEndCallbackRef = useRef<(() => void) | null>(null);

  const speak = useCallback((text: string, onEnd?: () => void) => {
    // Clear any existing speech
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Store the callback
    onEndCallbackRef.current = onEnd || null;

    // Set speaking state
    setIsSpeaking(true);
    setIsPaused(false);

    // Calculate speaking duration based on text length
    // Average speaking rate: ~150 words per minute = 2.5 words per second
    // Average word length: ~5 characters
    // So roughly: duration = (characters / 5) / 2.5 = characters / 12.5 seconds
    const words = text.split(/\s+/).length;
    const durationMs = (words / 2.5) * 1000; // 2.5 words per second

    // Log the speech for accessibility testing
    console.log(`[Mock TTS] Speaking (${Math.round(durationMs / 1000)}s): ${text.substring(0, 100)}${text.length > 100 ? '...' : ''}`);

    // Simulate speech duration
    timeoutRef.current = setTimeout(() => {
      setIsSpeaking(false);
      setIsPaused(false);
      onEndCallbackRef.current?.();
      onEndCallbackRef.current = null;
    }, durationMs);
  }, []);

  const pause = useCallback(() => {
    if (isSpeaking && !isPaused) {
      setIsPaused(true);
      console.log('[Mock TTS] Paused');
    }
  }, [isSpeaking, isPaused]);

  const resume = useCallback(() => {
    if (isPaused) {
      setIsPaused(false);
      console.log('[Mock TTS] Resumed');
    }
  }, [isPaused]);

  const stop = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setIsSpeaking(false);
    setIsPaused(false);
    onEndCallbackRef.current = null;
    console.log('[Mock TTS] Stopped');
  }, []);

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
