import { Router } from 'express';
import { getTTSClient, synthesize } from '../services/tts.js';

export const ttsRouter = Router();

/**
 * POST /api/tts
 * Body: { text?: string, ssml?: string, lang?: 'en-IN' | 'en-GB' }
 * Returns: { audio_base64: string, content_type: string }
 */
ttsRouter.post('/tts', async (req, res) => {
  try {
    const { text, ssml, lang = 'en-IN' } = req.body || {};
    const input = (ssml && ssml.trim()) ? { ssml: ssml.trim() } : (text && text.trim()) ? { text: text.trim() } : null;

    console.log('[TTS] POST /api/tts: lang=%s, hasText=%s, hasSsml=%s', lang, !!text, !!ssml);

    if (!input) {
      return res.status(400).json({
        detail: 'Request must include either "text" or "ssml" in the body.',
      });
    }

    const client = getTTSClient();
    if (!client) {
      console.error('[TTS] No Google TTS client — check GOOGLE_APPLICATION_CREDENTIALS');
      return res.status(503).json({
        detail: 'Text-to-Speech is not configured. Set GOOGLE_APPLICATION_CREDENTIALS or run with a Google Cloud identity.',
      });
    }

    const audioBuffer = await synthesize(client, { ...input, lang });
    const audioBase64 = audioBuffer.toString('base64');

    console.log('[TTS] Response: 200 OK, audio_base64 length=%d', audioBase64.length);
    res.json({
      audio_base64: audioBase64,
      content_type: 'audio/mp3',
    });
  } catch (err) {
    console.error('[TTS] Error:', err.message);
    const isConfig = /credential|auth|GOOGLE_APPLICATION_CREDENTIALS|permission|403|401/i.test(err.message || '');
    res.status(isConfig ? 503 : 500).json({
      detail: err.message || 'Failed to synthesize speech.',
    });
  }
});