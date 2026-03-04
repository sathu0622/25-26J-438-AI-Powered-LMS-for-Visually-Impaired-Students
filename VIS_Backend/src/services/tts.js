import textToSpeech from '@google-cloud/text-to-speech';

let client = null;

/**
 * Initialize Google TTS client using environment variables (Vercel-friendly)
 */
export function getTTSClient() {
  if (client) return client;

  const creds = {
    type: process.env.GOOGLE_TTS_TYPE,
    project_id: process.env.GOOGLE_TTS_PROJECT_ID,
    private_key_id: process.env.GOOGLE_TTS_PRIVATE_KEY_ID,
    private_key: process.env.GOOGLE_TTS_PRIVATE_KEY?.replace(/\\n/g, '\n'),
    client_email: process.env.GOOGLE_TTS_CLIENT_EMAIL,
    client_id: process.env.GOOGLE_TTS_CLIENT_ID,
    auth_uri: process.env.GOOGLE_TTS_AUTH_URI,
    token_uri: process.env.GOOGLE_TTS_TOKEN_URI,
    auth_provider_x509_cert_url: process.env.GOOGLE_TTS_AUTH_PROVIDER_CERT_URL,
    client_x509_cert_url: process.env.GOOGLE_TTS_CLIENT_CERT_URL,
  };

  console.log('[TTS] Initializing Google TTS client from ENV...');
  try {
    client = new textToSpeech.TextToSpeechClient({ credentials: creds });
    console.log('[TTS] Google TTS client initialized successfully.');
    return client;
  } catch (err) {
    console.error('[TTS] Failed to initialize Google TTS client:', err.message);
    return null;
  }
}

const VOICE_NAMES = {
  'en-IN': 'en-IN-Wavenet-A',
  'en-GB': 'en-GB-Wavenet-A',
};

export async function synthesize(ttsClient, { text, ssml, lang = 'en-IN' }) {
  const languageCode = lang === 'en-GB' ? 'en-GB' : 'en-IN';
  const voiceName = VOICE_NAMES[languageCode];

  console.log('[TTS] Synthesizing:', { languageCode, voiceName, input: ssml ? 'ssml' : 'text' });

  const request = {
    input: ssml ? { ssml } : { text: text || '' },
    voice: { languageCode, name: voiceName },
    audioConfig: { audioEncoding: 'MP3', speakingRate: 0.95, pitch: 0, volumeGainDb: 0 },
  };

  const [response] = await ttsClient.synthesizeSpeech(request);
  if (!response.audioContent) throw new Error('No audio content returned from TTS');

  return Buffer.from(response.audioContent);
}