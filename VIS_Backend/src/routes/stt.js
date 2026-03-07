import { Router } from 'express';
import { getSTTClient, transcribeWithHints, getPhraseHints, loadVocabulary, correctTranscript } from '../services/stt.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const VOCAB_PATH = path.join(__dirname, '../config/sinhala-vocabulary.json');

export const sttRouter = Router();

/**
 * POST /api/stt/transcribe
 * Transcribe audio with Sri Lankan historical phrase hints.
 * 
 * Body: { audio: base64-encoded audio data }
 * Optional query params: encoding, sampleRateHertz, languageCode
 */
sttRouter.post('/stt/transcribe', async (req, res) => {
  try {
    const sttClient = getSTTClient();
    if (!sttClient) {
      return res.status(503).json({
        error: 'Speech-to-Text service unavailable',
        message: 'Google Cloud credentials not configured',
      });
    }

    const { audio } = req.body;
    if (!audio) {
      return res.status(400).json({
        error: 'Missing audio data',
        message: 'Request body must contain "audio" field with base64-encoded audio',
      });
    }

    // Decode base64 audio
    const audioBuffer = Buffer.from(audio, 'base64');
    
    // Get options from query params or body
    const options = {
      encoding: req.body.encoding || req.query.encoding || 'WEBM_OPUS',
      sampleRateHertz: parseInt(req.body.sampleRateHertz || req.query.sampleRateHertz || '48000', 10),
      languageCode: req.body.languageCode || req.query.languageCode || 'en-LK',
    };

    const result = await transcribeWithHints(sttClient, audioBuffer, options);
    
    res.json({
      success: true,
      transcript: result.transcript,
      confidence: result.confidence,
      alternatives: result.alternatives,
    });
  } catch (error) {
    console.error('[STT Route] Error:', error.message);
    res.status(500).json({
      error: 'Transcription failed',
      message: error.message,
    });
  }
});

/**
 * GET /api/stt/hints
 * Get the list of phrase hints (for debugging/reference).
 */
sttRouter.get('/stt/hints', (req, res) => {
  res.json({
    hints: getPhraseHints(),
    count: getPhraseHints().length,
  });
});

/**
 * GET /api/stt/health
 * Check if STT service is available.
 */
sttRouter.get('/stt/health', (req, res) => {
  const sttClient = getSTTClient();
  res.json({
    status: sttClient ? 'ok' : 'unavailable',
    service: 'speech-to-text',
  });
});

// ========== VOCABULARY MANAGEMENT ENDPOINTS ==========

/**
 * GET /api/stt/vocabulary
 * Get the custom vocabulary for editing.
 */
sttRouter.get('/stt/vocabulary', (req, res) => {
  try {
    if (fs.existsSync(VOCAB_PATH)) {
      const data = fs.readFileSync(VOCAB_PATH, 'utf-8');
      const vocabulary = JSON.parse(data);
      res.json({
        success: true,
        vocabulary,
        path: VOCAB_PATH,
      });
    } else {
      res.status(404).json({
        success: false,
        error: 'Vocabulary file not found',
        path: VOCAB_PATH,
      });
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * POST /api/stt/vocabulary
 * Update the custom vocabulary and reload it.
 * Body: { corrections: {}, phraseHints: [] }
 */
sttRouter.post('/stt/vocabulary', (req, res) => {
  try {
    const { corrections, phraseHints } = req.body;
    
    // Load existing vocabulary
    let vocabulary = { corrections: {}, phraseHints: [] };
    if (fs.existsSync(VOCAB_PATH)) {
      vocabulary = JSON.parse(fs.readFileSync(VOCAB_PATH, 'utf-8'));
    }
    
    // Merge new corrections
    if (corrections) {
      vocabulary.corrections = { ...vocabulary.corrections, ...corrections };
    }
    
    // Update phrase hints if provided
    if (phraseHints) {
      // Combine and deduplicate
      const allHints = new Set([...vocabulary.phraseHints, ...phraseHints]);
      vocabulary.phraseHints = Array.from(allHints);
    }
    
    // Save
    fs.writeFileSync(VOCAB_PATH, JSON.stringify(vocabulary, null, 2), 'utf-8');
    
    // Reload vocabulary in STT service
    const loaded = loadVocabulary();
    
    res.json({
      success: true,
      message: 'Vocabulary updated and reloaded',
      corrections: Object.keys(loaded.corrections || {}).length,
      phraseHints: (loaded.phraseHints || []).length,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * POST /api/stt/vocabulary/add-correction
 * Add a single correction mapping.
 * Body: { word: "mihindu", misrecognitions: ["me hindu", "mi hindu"] }
 */
sttRouter.post('/stt/vocabulary/add-correction', (req, res) => {
  try {
    const { word, misrecognitions } = req.body;
    
    if (!word || !misrecognitions || !Array.isArray(misrecognitions)) {
      return res.status(400).json({
        success: false,
        error: 'Request must contain "word" (string) and "misrecognitions" (array)',
      });
    }
    
    // Load existing vocabulary
    let vocabulary = { corrections: {}, phraseHints: [] };
    if (fs.existsSync(VOCAB_PATH)) {
      vocabulary = JSON.parse(fs.readFileSync(VOCAB_PATH, 'utf-8'));
    }
    
    // Add or update correction
    const existing = vocabulary.corrections[word.toLowerCase()] || [];
    const merged = [...new Set([...existing, ...misrecognitions])];
    vocabulary.corrections[word.toLowerCase()] = merged;
    
    // Also add to phrase hints if not present
    if (!vocabulary.phraseHints.some(h => h.toLowerCase() === word.toLowerCase())) {
      vocabulary.phraseHints.push(word);
    }
    
    // Save
    fs.writeFileSync(VOCAB_PATH, JSON.stringify(vocabulary, null, 2), 'utf-8');
    
    // Reload
    loadVocabulary();
    
    res.json({
      success: true,
      message: `Added/updated correction for "${word}"`,
      word: word.toLowerCase(),
      misrecognitions: merged,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * POST /api/stt/vocabulary/reload
 * Reload vocabulary from the JSON file (after manual edits).
 */
sttRouter.post('/stt/vocabulary/reload', (req, res) => {
  try {
    const loaded = loadVocabulary();
    res.json({
      success: true,
      message: 'Vocabulary reloaded',
      corrections: Object.keys(loaded.corrections || {}).length,
      phraseHints: (loaded.phraseHints || []).length,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

/**
 * POST /api/stt/test-correction
 * Test the correction function on a transcript.
 * Body: { text: "me hindu the fifth" }
 */
sttRouter.post('/stt/test-correction', (req, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ success: false, error: 'Missing "text" field' });
    }
    
    const corrected = correctTranscript(text);
    res.json({
      success: true,
      original: text,
      corrected: corrected,
      changed: text !== corrected,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});
