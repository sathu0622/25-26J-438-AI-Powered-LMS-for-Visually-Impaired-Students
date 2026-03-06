/**
 * Central Text-to-Speech service for the app.
 * Prefer Google Cloud TTS (en-IN / en-GB) via backend when available;
 * fallback to browser Speech API with en-IN/en-GB, then mock for tests.
 * Optimized for visually impaired users: clear, consistent, natural speech.
 */

import { buildSsml, type SsmlOptions } from '../utils/ssml';

const API_BASE =
  (import.meta as any).env?.VITE_API_URL_VOICE || 'http://localhost:5000';
const USE_GOOGLE_TTS = (import.meta as any).env?.VITE_USE_GOOGLE_TTS === 'true';

export type TTSLang = 'en-IN' | 'en-GB';

export interface SpeakOptions {
  lang?: TTSLang;
  /** Speaking rate (0.25–4.0). Slightly slower for clarity. */
  rate?: number;
  /** Phrases to emphasize (SSML). */
  emphasis?: string[];
  /** If true, cancel current speech before starting. */
  interrupt?: boolean;
  /** Callback when playback ends. */
  onEnd?: () => void;
  /** Use SSML for this utterance (built from options). */
  ssmlOptions?: SsmlOptions;
}

type StateListener = (speaking: boolean) => void;
const stateListeners = new Set<StateListener>();
let currentSpeaking = false;

function setSpeaking(speaking: boolean) {
  if (currentSpeaking === speaking) return;
  currentSpeaking = speaking;
  stateListeners.forEach((l) => l(speaking));
}

export function isSpeaking(): boolean {
  return currentSpeaking;
}

export function addTTSStateListener(listener: StateListener): () => void {
  stateListeners.add(listener);
  return () => stateListeners.delete(listener);
}

/** Backend TTS: POST /api/tts, body { text?, ssml?, lang }, response { audio_base64, content_type }. */
async function fetchGoogleTTS(payload: {
  text?: string;
  ssml?: string;
  lang?: string;
}): Promise<{ audioBase64: string; contentType: string }> {
  const res = await fetch(`${API_BASE}/api/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`TTS API error: ${res.status}`);
  const data = await res.json();
  const audioBase64 = data.audio_base64 ?? data.audioBase64;
  const contentType = data.content_type ?? data.contentType ?? 'audio/mp3';
  if (!audioBase64) throw new Error('TTS API did not return audio');
  return { audioBase64, contentType };
}

let currentAudioRef: HTMLAudioElement | null = null;

/** Incremented on each speak() so only the latest request plays after async fetch. */
let speakId = 0;

function playBase64Audio(
  base64: string,
  contentType: string,
  onEnd: () => void
): void {
  // Stop any previous playback so voices don't overlap
  if (currentAudioRef) {
    try {
      currentAudioRef.pause();
      currentAudioRef.currentTime = 0;
    } catch {}
    currentAudioRef = null;
  }
  currentAudioRef = new Audio(`data:${contentType};base64,${base64}`);
  const audio = currentAudioRef;
  audio.onended = () => {
    if (currentAudioRef === audio) currentAudioRef = null;
    setSpeaking(false);
    onEnd();
  };
  audio.onerror = () => {
    if (currentAudioRef === audio) currentAudioRef = null;
    setSpeaking(false);
    onEnd();
  };
  setSpeaking(true);
  audio.play().catch(() => {
    if (currentAudioRef === audio) currentAudioRef = null;
    setSpeaking(false);
    onEnd();
  });
}
let currentUtteranceRef: SpeechSynthesisUtterance | null = null;

function speakWithBrowser(
  text: string,
  lang: TTSLang,
  rate: number,
  onEnd: () => void
): void {
  if (!('speechSynthesis' in window)) {
    onEnd();
    return;
  }
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang;
  u.rate = rate;
  u.pitch = 1;
  u.volume = 1;
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find((v) => v.lang === lang) ?? voices.find((v) => v.lang.startsWith('en'));
  if (preferred) u.voice = preferred;
  u.onend = () => {
    currentUtteranceRef = null;
    setSpeaking(false);
    onEnd();
  };
  u.onerror = () => {
    currentUtteranceRef = null;
    setSpeaking(false);
    onEnd();
  };
  currentUtteranceRef = u;
  setSpeaking(true);
  window.speechSynthesis.speak(u);
}

let mockTimeoutId: ReturnType<typeof setTimeout> | null = null;

function speakWithMock(text: string, onEnd: () => void): void {
  if (mockTimeoutId) clearTimeout(mockTimeoutId);
  const words = text.split(/\s+/).length;
  const durationMs = Math.max(1000, (words / 2.5) * 1000);
  setSpeaking(true);
  mockTimeoutId = setTimeout(() => {
    mockTimeoutId = null;
    setSpeaking(false);
    onEnd();
  }, durationMs);
}

/**
 * Central TTS: speak text with optional SSML tuning (pronunciation, speed, emphasis).
 * Uses Google Cloud TTS via backend when VITE_USE_GOOGLE_TTS=true and /api/tts is available;
 * otherwise browser (en-IN/en-GB) or mock.
 */
export async function speak(text: string, options: SpeakOptions = {}): Promise<void> {
  const lang: TTSLang = options.lang ?? 'en-IN';
  const rate = options.rate ?? 0.95;
  const interrupt = options.interrupt !== false;
  const onEnd = options.onEnd ?? (() => {});

  if (interrupt) cancel();

  const trimmed = text?.trim();
  if (!trimmed) {
    onEnd();
    return;
  }

  const doOnEnd = () => {
    onEnd();
  };

  // 1) Try Google TTS via backend
  if (USE_GOOGLE_TTS) {
    try {
      const mySpeakId = ++speakId;
      const ssmlOptions: SsmlOptions = {
        rate,
        emphasis: options.emphasis,
        ...options.ssmlOptions,
      };
      const ssml = buildSsml(trimmed, ssmlOptions);
      const { audioBase64, contentType } = await fetchGoogleTTS({ ssml, lang });
      // Only play if no newer speak() was called (avoid overlapping after race)
      if (mySpeakId !== speakId) return;
      playBase64Audio(audioBase64, contentType, doOnEnd);
      return;
    } catch (e) {
      console.warn('Google TTS unavailable, using browser/mock:', e);
    }
  }

  // 2) Browser Speech API (en-IN / en-GB)
  if ('speechSynthesis' in window) {
    speakWithBrowser(trimmed, lang, rate, doOnEnd);
    return;
  }

  // 3) Mock for tests/demos
  speakWithMock(trimmed, doOnEnd);
}

/**
 * Cancel any current TTS (all backends).
 */
export function cancel(): void {
  setSpeaking(false);
  if (mockTimeoutId) {
    clearTimeout(mockTimeoutId);
    mockTimeoutId = null;
  }
  if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  if (currentAudioRef) {
    try {
      currentAudioRef.pause();
    } catch {}
    currentAudioRef = null;
  }
}

/**
 * Speak a short announcement (e.g. navigation, button). Uses consistent rate and brief pause.
 */
export function announce(phrase: string, options: SpeakOptions = {}): Promise<void> {
  return speak(phrase, { ...options, rate: options.rate ?? 0.95, interrupt: true });
}