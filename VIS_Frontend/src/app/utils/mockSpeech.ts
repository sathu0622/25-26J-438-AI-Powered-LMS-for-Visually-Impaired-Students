/**
 * Mock speech synthesis utility
 * Simulates speech without requiring browser APIs
 * Perfect for demos and testing accessibility features
 */

interface MockUtterance {
  text: string;
  onEnd?: () => void;
  timeoutId?: NodeJS.Timeout;
}

let currentUtterance: MockUtterance | null = null;

/**
 * Calculate speaking duration based on text length
 * Average speaking rate: ~150 words per minute = 2.5 words per second
 */
const calculateDuration = (text: string): number => {
  const words = text.split(/\s+/).length;
  const durationMs = (words / 2.5) * 1000; // 2.5 words per second
  return Math.max(1000, durationMs); // Minimum 1 second
};

/**
 * Safely speak text using mock synthesis
 * @param text - The text to speak
 * @param onEnd - Optional callback when speech ends
 * @returns A mock utterance object
 */
export const safeSpeak = async (
  text: string,
  onEnd?: () => void
): Promise<MockUtterance | null> => {
  // Cancel any ongoing speech
  safeCancel();

  const duration = calculateDuration(text);
  
  // Log for accessibility testing
  const preview = text.length > 100 ? `${text.substring(0, 100)}...` : text;
  console.log(`[Mock TTS] Speaking (${Math.round(duration / 1000)}s): ${preview}`);

  const utterance: MockUtterance = {
    text,
    onEnd,
  };

  // Simulate speech duration
  utterance.timeoutId = setTimeout(() => {
    currentUtterance = null;
    onEnd?.();
  }, duration);

  currentUtterance = utterance;
  return utterance;
};

/**
 * Safely cancel ongoing speech
 */
export const safeCancel = () => {
  if (currentUtterance?.timeoutId) {
    clearTimeout(currentUtterance.timeoutId);
    currentUtterance = null;
    console.log('[Mock TTS] Cancelled');
  }
};

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
