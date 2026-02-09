/**
 * Safe speech synthesis utility
 * Handles browser compatibility, voice loading, and errors gracefully
 */

let voicesLoaded = false;
let voiceLoadPromise: Promise<void> | null = null;
let hasCheckedSupport = false;
let isActuallySupported = false;

// Check if speech synthesis is actually supported
const checkSupport = (): boolean => {
  if (hasCheckedSupport) return isActuallySupported;
  
  hasCheckedSupport = true;
  
  if (typeof window === 'undefined') {
    isActuallySupported = false;
    return false;
  }
  
  // Check for speech synthesis support
  if (!('speechSynthesis' in window)) {
    console.warn('Speech synthesis not available in this browser');
    isActuallySupported = false;
    return false;
  }
  
  // Check if it's a real implementation or a stub
  try {
    const utterance = new SpeechSynthesisUtterance('');
    if (!utterance) {
      isActuallySupported = false;
      return false;
    }
    isActuallySupported = true;
    return true;
  } catch (error) {
    console.warn('Speech synthesis initialization failed:', error);
    isActuallySupported = false;
    return false;
  }
};

// Initialize voice loading
const loadVoices = (): Promise<void> => {
  if (!checkSupport()) {
    return Promise.resolve();
  }
  
  if (voiceLoadPromise) return voiceLoadPromise;

  voiceLoadPromise = new Promise((resolve) => {
    const checkVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        voicesLoaded = true;
        resolve();
      }
    };

    // Check immediately
    checkVoices();

    // Listen for voices changed event (Chrome)
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = () => {
        checkVoices();
      };
    }

    // Fallback timeout
    setTimeout(() => {
      voicesLoaded = true;
      resolve();
    }, 1000);
  });

  return voiceLoadPromise;
};

/**
 * Safely speak text using Web Speech API
 * @param text - The text to speak
 * @param onEnd - Optional callback when speech ends
 * @returns The utterance object or null if failed
 */
export const safeSpeak = async (
  text: string,
  onEnd?: () => void
): Promise<SpeechSynthesisUtterance | null> => {
  // Check browser support
  if (!checkSupport()) {
    console.warn('Speech synthesis not supported');
    onEnd?.();
    return null;
  }

  try {
    // Wait for voices to load
    await loadVoices();

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    // Small delay after cancel to ensure it completes
    await new Promise(resolve => setTimeout(resolve, 50));

    const utterance = new SpeechSynthesisUtterance(text);

    // Get available voices and prefer English
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find(voice => voice.lang.startsWith('en'));
    if (englishVoice) {
      utterance.voice = englishVoice;
    }

    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;
    utterance.lang = 'en-US';

    utterance.onend = () => {
      onEnd?.();
    };

    utterance.onerror = (event) => {
      // Suppress expected errors
      if (event.error !== 'interrupted' && event.error !== 'canceled') {
        console.error('Speech error:', event.error);
      }
      onEnd?.();
    };

    window.speechSynthesis.speak(utterance);
    return utterance;
  } catch (error) {
    console.error('Speech synthesis error:', error);
    onEnd?.();
    return null;
  }
};

/**
 * Safely cancel ongoing speech
 */
export const safeCancel = () => {
  if ('speechSynthesis' in window) {
    try {
      window.speechSynthesis.cancel();
    } catch (error) {
      console.error('Error canceling speech:', error);
    }
  }
};

/**
 * Check if speech synthesis is supported
 */
export const isSpeechSupported = (): boolean => {
  return checkSupport();
};

// Preload voices on module load
if (typeof window !== 'undefined') {
  loadVoices();
}