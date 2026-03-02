import textToSpeech from '@google-cloud/text-to-speech';

let client = null;

/**
 * Lazy-init TTS client. Requires GOOGLE_APPLICATION_CREDENTIALS env pointing to a service account JSON,
 * or running on GCP (e.g. Cloud Run) with a default identity.
 */
export function getTTSClient() {
  if (client !== null) return client;
  const credsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
  console.log('[TTS] Initializing Google TTS client. GOOGLE_APPLICATION_CREDENTIALS:', credsPath || '(not set)');
  try {
    client = new textToSpeech.TextToSpeechClient();
    console.log('[TTS] Google TTS client initialized successfully.');
    return client;
  } catch (e) {
    console.warn('[TTS] Google TTS client init failed:', e.message);
    return null;
  }
}

/** Voice names for en-IN and en-GB (Wavenet for natural speech). */
const VOICE_NAMES = {
  'en-IN': 'en-IN-Wavenet-A',
  'en-GB': 'en-GB-Wavenet-A',
};

/**
 * Synthesize speech from text or SSML.
 * @param {import('@google-cloud/text-to-speech').TextToSpeechClient} ttsClient
 * @param {{ text?: string, ssml?: string, lang?: string }} options
 * @returns {Promise<Buffer>} Raw MP3 audio buffer
 */
export async function synthesize(ttsClient, { text, ssml, lang = 'en-IN' }) {
  const languageCode = lang === 'en-GB' ? 'en-GB' : 'en-IN';
  const voiceName = VOICE_NAMES[languageCode] || VOICE_NAMES['en-IN'];

  console.log('[TTS] Synthesizing: lang=%s, voice=%s, input=%s', languageCode, voiceName, ssml ? 'ssml' : 'text');
  const request = {
    input: ssml ? { ssml } : { text: text || '' },
    voice: {
      languageCode,
      name: voiceName,
    },
    audioConfig: {
      audioEncoding: 'MP3',
      speakingRate: 0.95,
      pitch: 0,
      volumeGainDb: 0,
    },
  };

  const [response] = await ttsClient.synthesizeSpeech(request);
  if (!response.audioContent) {
    throw new Error('No audio content in TTS response');
  }
  const buffer = Buffer.from(response.audioContent);
  console.log('[TTS] Synthesis OK, audio size:', buffer.length, 'bytes');
  return buffer;
}
