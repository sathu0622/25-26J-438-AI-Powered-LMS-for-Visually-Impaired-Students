/**
 * TTS Context: provides central speak/cancel and isSpeaking to the whole app.
 * Use for automatic, clear, consistent voice feedback (en-IN / en-GB, SSML-tuned).
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import * as tts from '../services/ttsService';
import type { SpeakOptions } from '../services/ttsService';

interface TTSContextValue {
  speak: (text: string, options?: SpeakOptions) => Promise<void>;
  cancel: () => void;
  announce: (phrase: string, options?: SpeakOptions) => Promise<void>;
  isSpeaking: boolean;
}

const TTSContext = createContext<TTSContextValue | null>(null);

export function TTSProvider({ children }: { children: ReactNode }) {
  const [isSpeaking, setIsSpeaking] = useState(tts.isSpeaking());

  useEffect(() => {
    const unsub = tts.addTTSStateListener(setIsSpeaking);
    return unsub;
  }, []);

  const speak = useCallback((text: string, options?: SpeakOptions) => {
    return tts.speak(text, options);
  }, []);

  const cancel = useCallback(() => {
    tts.cancel();
  }, []);

  const announce = useCallback((phrase: string, options?: SpeakOptions) => {
    return tts.announce(phrase, options);
  }, []);

  const value: TTSContextValue = {
    speak,
    cancel,
    announce,
    isSpeaking,
  };

  return <TTSContext.Provider value={value}>{children}</TTSContext.Provider>;
}

export function useTTS(): TTSContextValue {
  const ctx = useContext(TTSContext);
  if (!ctx) {
    throw new Error('useTTS must be used within TTSProvider');
  }
  return ctx;
}

/** Optional hook: returns same API but no-op if outside provider (for components that may render without provider). */
export function useTTSSafe(): TTSContextValue {
  const ctx = useContext(TTSContext);
  if (!ctx) {
    return {
      speak: async () => {},
      cancel: () => {},
      announce: async () => {},
      isSpeaking: false,
    };
  }
  return ctx;
}