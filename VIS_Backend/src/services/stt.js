import speech from '@google-cloud/speech';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let client = null;

// ========== LOAD CUSTOM VOCABULARY ==========
const VOCAB_PATH = path.join(__dirname, '../config/sinhala-vocabulary.json');
let customVocabulary = { corrections: {}, phraseHints: [] };

/**
 * Load or reload the custom Sinhala vocabulary from JSON file.
 * Call this to pick up changes without restarting the server.
 */
export function loadVocabulary() {
  try {
    if (fs.existsSync(VOCAB_PATH)) {
      const data = fs.readFileSync(VOCAB_PATH, 'utf-8');
      customVocabulary = JSON.parse(data);
      console.log('[STT] Loaded vocabulary: %d corrections, %d phrase hints',
        Object.keys(customVocabulary.corrections || {}).length,
        (customVocabulary.phraseHints || []).length);
    } else {
      console.warn('[STT] Vocabulary file not found:', VOCAB_PATH);
    }
  } catch (e) {
    console.error('[STT] Error loading vocabulary:', e.message);
  }
  return customVocabulary;
}

// Load vocabulary on module init
loadVocabulary();

/**
 * Apply custom corrections to transcript based on vocabulary.
 * This corrects misrecognized Sinhala words.
 */
export function correctTranscript(transcript) {
  if (!transcript || !customVocabulary.corrections) return transcript;
  
  let corrected = transcript.toLowerCase();
  
  for (const [correct, variants] of Object.entries(customVocabulary.corrections)) {
    for (const variant of variants) {
      // Use word boundary regex to avoid partial replacements
      const regex = new RegExp(`\\b${variant.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
      corrected = corrected.replace(regex, correct);
    }
  }
  
  return corrected;
}

/**
 * Lazy-init Speech-to-Text client.
 */
export function getSTTClient() {
  if (client !== null) return client;
  const credsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
  console.log('[STT] Initializing Google STT client. GOOGLE_APPLICATION_CREDENTIALS:', credsPath || '(not set)');
  try {
    client = new speech.SpeechClient();
    console.log('[STT] Google STT client initialized successfully.');
    return client;
  } catch (e) {
    console.warn('[STT] Google STT client init failed:', e.message);
    return null;
  }
}

/**
 * Sri Lankan historical vocabulary for phrase hints.
 * These words will be prioritized during transcription.
 */
const SINHALA_PHRASE_HINTS = [
  // Places
  'Anuradhapura', 'Polonnaruwa', 'Sigiriya', 'Kandy', 'Dambulla', 
  'Mihintale', 'Nuwara Eliya', 'Trincomalee', 'Jaffna', 'Galle',
  'Matara', 'Ratnapura', 'Kurunegala', 'Batticaloa', 'Gampola',
  'Kotte', 'Talawakele', 'Kuruwita', 'Pahiyangala', 'Batadombalena',
  'Lankapatuna', 'Lankapattana', 'Maduru Oya', 'Minihagalkanda',
  'Randeniwela', 'Panakaduwa', 'Diksanda', 'Serendib', 'Sarandib',
  
  // Historical Kings & Figures
  'Dutugemunu', 'Parakramabahu', 'Vijaya', 'Vijayabahu',
  'Devanampiyatissa', 'Kasyapa', 'Kashyapa', 'Elara', 
  'Nissankamalla', 'Keerthi Sri Nissankamalla', 'Valagamba',
  'Pandukabhaya', 'Vasabha', 'Mahasen', 'Mahasena', 'Mihindu',
  'Mahinda', 'Sena Sammatha Wickramabahu', 'Pararajasekaram',
  'Leelawathie', 'Lilavathi', 'Parakramabahu the Great',
  
  // Modern Historical Figures
  'Arumuga Navalar', 'Walisingha Harischandra', 
  'Ponnambalam Arunachalam', 'M.C. Siddhi Lebbe', 'Siddhi Lebbe',
  'C.W.W. Kannangara', 'Kannangara', 'D.S. Senanayake', 'Senanayake',
  'Henry Steel Olcott', 'Olcott', 'John Doyly',
  
  // Literary Sources
  'Mahavamsa', 'Dipavamsa', 'Deepavamsa', 'Vamsatthappakasini',
  'Sandesha Kavya', 'Prasasti Kavya', 'Hatan Kavya', 'Masika Thegga',
  'Brahmi', 'Giri Lipi',
  
  // Historical Terms
  'Dagoba', 'Vihara', 'Viharaya', 'Samadhi', 'Sangha', 'Buddhism',
  'Chakravarthi', 'Parumaka', 'Gamika', 'Olagam', 'Pattanagama',
  'Paravenigam', 'Badde', 'Nayakkar',
  
  // Monuments & Architecture  
  'Ruwanweliseya', 'Ruwanweli Seya', 'Jetavanarama', 'Jetavana',
  'Abhayagiri', 'Thuparama', 'Lovamahapaya', 'Lohamahapaya',
  'Isurumuniya', 'Aukana', 'Avukana', 'Kuttam Pokuna',
  'Gauta Pillars', 'Gauto Kanu',
  
  // Technical Terms
  'Sakaporuwa', 'Potters wheel', 'Kahapana', 'Sluice Gate',
  'Ralapanawa', 'Uraketa Lin',
  
  // Taxes & Administrative
  'Kethi Ada', 'Dakapathi', 'Gruhadanda', 'Madige Badda', 'Vejjasala',
  
  // Rivers & Geography
  'Mahaweli', 'Mahaweli River', 'Amban River',
  
  // Armies & Groups
  'Welayikkar', 'Welayikkar Army', 'Samorin',
  
  // Institutions & Events
  'Seneviya Pirivena', 'Zahira College', 'Ceylon National Congress',
  'Jothiya Sitana', 'Kandyan Convention',
  
  // Foreign Names in Sri Lankan Context
  'Cheng Ho', 'Zheng He', 'Vasco de Gama', 'Constantinu de Sa',
  
  // Common Historical Terms
  'cinnamon', 'spices', 'irrigation', 'tank', 'monastery',
  'inscription', 'coins', 'copper plate', 'slab inscription',
  'prehistoric', 'historic era', 'Anuradhapura period',
  'Polonnaruwa period', 'Kandyan period', 'colonial',
  
  // Numbers (Cardinal) - for king names like "Mihindu 5"
  'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
  'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
  
  // Numbers (Ordinal) - for king names like "Mihindu the fifth"
  'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
  'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth',
  'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth',
  
  // Roman numerals (spoken as letters)
  'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
  'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
  
  // Common king name combinations
  'Mihindu V', 'Mihindu five', 'Mihindu fifth', 'Mihindu the fifth',
  'Parakramabahu I', 'Parakramabahu II', 'Parakramabahu the first', 'Parakramabahu the second',
  'Vijayabahu I', 'Vijayabahu II', 'Vijayabahu III', 'Vijayabahu the first',
  'Nissankamalla I', 'Nissankamalla the first',
  'Dutugemunu I', 'Dutugemunu the first',
];

/**
 * Transcribe audio using Google Cloud Speech-to-Text with phrase hints.
 * @param {speech.SpeechClient} sttClient
 * @param {Buffer} audioBuffer - Raw audio data
 * @param {object} options
 * @param {string} [options.encoding='WEBM_OPUS'] - Audio encoding
 * @param {number} [options.sampleRateHertz=48000] - Sample rate
 * @param {string} [options.languageCode='en-LK'] - Language code
 * @returns {Promise<{transcript: string, confidence: number}>}
 */
export async function transcribeWithHints(sttClient, audioBuffer, options = {}) {
  const {
    encoding = 'WEBM_OPUS',
    sampleRateHertz = 48000,
    languageCode = 'en-LK',
  } = options;

  console.log('[STT] Transcribing: encoding=%s, sampleRate=%d, lang=%s, audioSize=%d bytes',
    encoding, sampleRateHertz, languageCode, audioBuffer.length);

  const request = {
    audio: {
      content: audioBuffer.toString('base64'),
    },
    config: {
      encoding,
      sampleRateHertz,
      languageCode,
      // Alternative languages for better recognition
      alternativeLanguageCodes: ['en-US', 'en-IN'],
      // Enable automatic punctuation
      enableAutomaticPunctuation: true,
      // Model optimized for short queries
      model: 'default',
      // Use enhanced model for better accuracy
      useEnhanced: true,
      // Speech adaptation with phrase hints - THIS IS THE KEY
      // Combine hardcoded hints with custom vocabulary hints
      adaptation: {
        phraseSets: [
          {
            phrases: [
              ...SINHALA_PHRASE_HINTS,
              ...(customVocabulary.phraseHints || []),
            ].map(phrase => ({
              value: phrase,
              boost: 15, // Strong boost for these phrases (range: -20 to 20)
            })),
          },
        ],
      },
      // Request multiple alternatives
      maxAlternatives: 3,
    },
  };

  try {
    const [response] = await sttClient.recognize(request);
    
    if (!response.results || response.results.length === 0) {
      console.log('[STT] No transcription results');
      return { transcript: '', confidence: 0 };
    }

    // Get the best result
    const result = response.results[0];
    const alternative = result.alternatives[0];
    const rawTranscript = alternative.transcript || '';
    
    // Apply custom vocabulary corrections
    const correctedTranscript = correctTranscript(rawTranscript);
    
    console.log('[STT] Raw: "%s"', rawTranscript);
    console.log('[STT] Corrected: "%s" (confidence: %d%%)',
      correctedTranscript, Math.round((alternative.confidence || 0) * 100));
    
    // Log alternatives for debugging
    if (result.alternatives.length > 1) {
      console.log('[STT] Alternatives:', result.alternatives.slice(1).map(a => a.transcript));
    }

    return {
      transcript: correctedTranscript,
      rawTranscript: rawTranscript,
      confidence: alternative.confidence || 0,
      alternatives: result.alternatives.map(a => ({
        transcript: correctTranscript(a.transcript),
        rawTranscript: a.transcript,
        confidence: a.confidence,
      })),
    };
  } catch (error) {
    console.error('[STT] Transcription error:', error.message);
    throw error;
  }
}

/**
 * Get the list of phrase hints (useful for frontend display or debugging)
 */
export function getPhraseHints() {
  return SINHALA_PHRASE_HINTS;
}
