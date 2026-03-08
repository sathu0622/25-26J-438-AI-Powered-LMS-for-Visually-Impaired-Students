/**
 * Mock speech synthesis utility
 * Simulates speech without requiring browser APIs
 * Perfect for demos and testing accessibility features
 */

interface MockUtterance {
  text: string;
  onEnd?: () => void;
  timeoutId?: ReturnType<typeof setTimeout>;
}

let currentUtterance: MockUtterance | null = null;

/**
 * Calculate speaking duration based on text length
 * Average speaking rate: ~150 words per minute = 2.5 words per second
 */
function isSpeechSynthesisSupported() {
  return typeof window !== 'undefined' && 'speechSynthesis' in window && typeof window.SpeechSynthesisUtterance === 'function';
}

export function safeSpeak(text: string, onEnd?: () => void) {
  safeCancel();
  if (!text) return;
  const duration = Math.max(1, Math.min(8, Math.round(text.length / 30)));
  console.log(`[Mock TTS] Speaking (${duration}s): ${text}`);
  if (isSpeechSynthesisSupported()) {
    try {
      const utter = new window.SpeechSynthesisUtterance(text);
      if (onEnd) utter.onend = onEnd;
      window.speechSynthesis.speak(utter);
    } catch (err) {
      console.warn('Speech synthesis error:', err);
      alert('Speech synthesis failed. Please check your browser settings.');
      if (onEnd) onEnd();
    }
  } else {
    console.warn('Speech synthesis not supported in this browser');
    alert('Speech synthesis is not supported in your browser. Please use Chrome, Edge, or Firefox in normal mode.');
    if (onEnd) onEnd();
  }
}

export function safeCancel() {
  if (isSpeechSynthesisSupported()) {
    window.speechSynthesis.cancel();
    // console.log('[Mock TTS] Cancelled');
  }
}

/**
 * Check if speech synthesis is supported (always true for mock)
 */
export const isSpeechSupported = (): boolean => {
  return true;
};

/**
 * Check if currently speaking
 */
export const isSpeaking = (): boolean => {
  return currentUtterance !== null;
};
