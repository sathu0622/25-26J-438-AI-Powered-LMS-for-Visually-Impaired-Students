import React, { useState, useRef } from 'react';

// Sinhala/Sri Lankan place names and historical terms that are commonly misrecognized
const SINHALA_VOCABULARY: Record<string, string[]> = {
  // ===== PLACES =====
  'anuradhapura': ['anuradapura', 'anura dapura', 'anura dhapura', 'anooradhapura', 'anura the pura', 'anurada pura'],
  'polonnaruwa': ['polonaruwa', 'polan naruwa', 'polo naruwa', 'pollonnaruwa', 'pollonaruwa', 'polon naruwa'],
  'sigiriya': ['sigiri', 'sigirya', 'sigiri ya', 'see giriya', 'seegiriya', 'sigi riya'],
  'kandy': ['candy', 'candi', 'kandie', 'kandi'],
  'dambulla': ['dambula', 'dam bulla', 'damboola', 'dambola', 'dam bula'],
  'mihintale': ['mi hintale', 'mihinthale', 'mee hintale', 'me hintale', 'mihin tale'],
  'nuwara eliya': ['noora eliya', 'nuwara elia', 'nuvara eliya', 'nuwara elia'],
  'trincomalee': ['trinco', 'trincomalee', 'trinco mali', 'trinco malee'],
  'jaffna': ['yalpana', 'japana', 'jafna', 'yalpanam'],
  'galle': ['gal', 'gal le', 'gaul'],
  'matara': ['mathara', 'mathra', 'ma thara'],
  'ratnapura': ['rathnapura', 'ratna pura', 'rathna pura', 'ratna poora'],
  'kurunegala': ['kurunagala', 'kurune gala', 'kurunae gala'],
  'batticaloa': ['batti', 'baticaloa', 'batti caloa', 'batticaloa'],
  'gampola': ['gam pola', 'gampole', 'gam pole'],
  'kotte': ['cotte', 'kot te', 'kottay'],
  'talawakele': ['talawakale', 'talawa kele', 'thalawakele'],
  'kuruwita': ['kuruvita', 'kuru wita', 'kuruwitha'],
  'pahiyangala': ['pahiyan gala', 'pahi yangala', 'pahiyangale', 'pahiyan gale'],
  'batadombalena': ['batadamba lena', 'batadom balena', 'bata domba lena'],
  'lankapatuna': ['lankapattana', 'lanka patuna', 'lanka pattana', 'lankapathuna'],
  'maduru oya': ['maduru-oya', 'maduru oya', 'madhuru oya', 'maduruoya'],
  'minihagalkanda': ['miniha galkanda', 'mini hagal kanda', 'minihagal kanda'],
  'randeniwela': ['randeni wela', 'randenivela', 'rande niwela'],
  'panakaduwa': ['panaka duwa', 'pana kaduwa', 'panaka duva'],
  'diksanda': ['dik sanda', 'diksha nda', 'diksanda'],
  'serendib': ['sarandib', 'seran dib', 'saran dib', 'serendip'],
  'istanbul': ['constantinople', 'constant inople', 'constan tinople'],
  
  // ===== HISTORICAL KINGS & FIGURES =====
  'dutugemunu': ['dutugamunu', 'dutu gemunu', 'duttu gemunu', 'dutagemunu', 'dutuge munu'],
  'parakramabahu': ['parakrama bahu', 'prakramabahu', 'parakrama', 'parakrambahu'],
  'vijaya': ['vijay', 'vee jaya', 'vijayaa', 'wijaya'],
  'vijayabahu': ['vijaya bahu', 'wijayabahu', 'vijayabaahu'],
  'devanampiyatissa': ['devanampiya tissa', 'devanam piyatissa', 'devanampiya', 'devanam piya tissa'],
  'kasyapa': ['kashyapa', 'kasyap', 'kashyap', 'cassapa', 'kaasyapa'],
  'elara': ['ellara', 'elala', 'ela ra', 'eelara'],
  'nissankamalla': ['nissanka malla', 'nishanka malla', 'keerthi sri nissankamalla', 'keerthi nissanka malla'],
  'valagamba': ['walagamba', 'wala gamba', 'valagambahu', 'wala gambahu'],
  'pandukabhaya': ['panduka bhaya', 'pandu kabhaya', 'pandukabaya', 'pandu ka bhaya'],
  'vasabha': ['wasabha', 'vasa bha', 'wasa bha'],
  'mahasen': ['maha sen', 'mahasena', 'maha sena'],
  'mihindu': ['mahinda', 'mi hindu', 'me hindu', 'mahindha'],
  'sena sammatha wickramabahu': ['sena sammatha', 'wickramabahu', 'sena sam matha', 'vikramabahu'],
  'pararajasekaram': ['para raja sekaram', 'pararajasekaran', 'para rajasekaram'],
  'leelawathie': ['leelawathi', 'leela vathie', 'lilavathi', 'leelavathi'],
  
  // ===== MODERN HISTORICAL FIGURES =====
  'arumuga navalar': ['arumugam navalar', 'arumuga nawalar', 'arumugha navalar'],
  'walisingha harischandra': ['walisinghe harischandra', 'valisingha harischandra', 'walisinghe harischndra'],
  'ponnambalam arunachalam': ['ponnambalam', 'arunachalam', 'pon nambalam'],
  'siddhi lebbe': ['siddi lebbe', 'siddilebbe', 'siddi lebe', 'sidi lebbe'],
  'kannangara': ['kannan gara', 'kanan gara', 'c.w.w. kannangara'],
  'senanayake': ['senanayaka', 'sena nayake', 'd.s. senanayake'],
  'olcott': ['alcott', 'ol cott', 'henry steel olcott'],
  
  // ===== LITERARY SOURCES =====
  'mahavamsa': ['maha vamsa', 'mahawamsa', 'maha wamsa', 'mahavansa'],
  'dipavamsa': ['dipa vamsa', 'deepa vamsa', 'dipawamsa', 'deepavamsa', 'deepawamsa'],
  'vamsatthappakasini': ['vamsattha ppakasini', 'vamsathapakasini', 'vamsatta pakasini'],
  'sandesha kavya': ['sandesya kavya', 'sandesa kavya', 'sandesha kavya'],
  'prasasti kavya': ['prasasthi kavya', 'prasasti kavya', 'pra sasti kavya'],
  'hatan kavya': ['hatana kavya', 'hatan kavya', 'hatankavya'],
  'masika thegga': ['masika tegga', 'masika thega', 'maasika thegga'],
  
  // ===== HISTORICAL TERMS =====
  'dagoba': ['dagaba', 'dageba', 'stupa', 'da goba'],
  'vihara': ['viharaya', 'vihare', 'vi hara', 'vihaara'],
  'samadhi': ['samaadhi', 'sama dhi', 'samadi'],
  'sangha': ['sanga', 'sang ha', 'sangaa'],
  'brahmi': ['brahmy', 'bra mi', 'brami', 'braahmi'],
  'giri lipi': ['girilipi', 'giri lipy', 'giri lipee'],
  'chakravarthi': ['chakravarthy', 'chakra varthi', 'sakravarti', 'chakrawarti'],
  'parumaka': ['paru maka', 'parumakha', 'parumacka'],
  'gamika': ['gami ka', 'gamikha', 'gameeka'],
  'olagam': ['olag am', 'olagem', 'ola gam'],
  'pattanagama': ['pattana gama', 'patanagama', 'patan gama'],
  'paravenigam': ['paraveni gam', 'paraveni kam', 'paraveniya'],
  'badde': ['badda', 'bad de', 'badday'],
  
  // ===== MONUMENTS & ARCHITECTURE =====
  'ruwanweliseya': ['ruwanweli', 'ruwan weli', 'ruwanwelisaya', 'ruvanveli', 'ruwanweli seya'],
  'jetavanarama': ['jetavana', 'jetawana', 'jeta vanarama', 'jetavana rama'],
  'abhayagiri': ['abhaya giri', 'abhayagiriya', 'abaya giri'],
  'thuparama': ['thupa rama', 'thuparamaya', 'toopa rama'],
  'lovamahapaya': ['lova maha paya', 'loha mahapaya', 'lova mahapaya', 'lohamahapaya'],
  'isurumuniya': ['isuru muniya', 'isurumuni', 'isurumuiya', 'isuru muni'],
  'aukana': ['awkana', 'au kana', 'auka na', 'avukana'],
  'kuttam pokuna': ['kuttam pokuna', 'kutam pokuna', 'kuttam pokhuna', 'twin ponds'],
  'gauta pillars': ['gauto kanu', 'gauta kanu', 'gauta pillar', 'gawta pillars'],
  
  // ===== TECHNICAL TERMS =====
  'sakaporuwa': ['saka poruwa', 'sakaporuva', 'potters wheel', 'sakha poruwa'],
  'kahapana': ['kaha pana', 'kahapanaya', 'kaha panaya'],
  'sluice gate': ['sluice', 'sloo ice gate', 'sluce gate'],
  'ralapanawa': ['rala panawa', 'ralapanava', 'rala panava'],
  'uraketa lin': ['uraketa', 'uraketa lin', 'ura keta lin'],
  
  // ===== TAXES & ADMINISTRATIVE =====
  'kethi ada': ['kethi adha', 'keth ada', 'kethiyada'],
  'dakapathi': ['daka pathi', 'dakapati', 'daka pati'],
  'gruhadanda': ['gruha danda', 'gruhadan da', 'griha danda'],
  'madige badda': ['madige bada', 'madi ge badda', 'madigae badda'],
  'vejjasala': ['vejja sala', 'vejja saala', 'vaidya sala'],
  
  // ===== RIVERS & GEOGRAPHY =====
  'mahaweli': ['maha weli', 'mahaweli', 'mahaveli', 'maha veli'],
  'amban river': ['amban', 'ambhan', 'am ban river'],
  
  // ===== ARMIES & GROUPS =====
  'welayikkar': ['welayikkar army', 'velai ikkar', 'welaikkar', 'velaikkar'],
  'samorin': ['saa morin', 'samoorin', 'zamorin'],
  
  // ===== INSTITUTIONS =====
  'seneviya pirivena': ['senevi pirivena', 'sena viya pirivena', 'seneviya'],
  'zahira college': ['zahira', 'zahera college', 'zahira collage'],
  'ceylon national congress': ['ceylon congress', 'national congress'],
  
  // ===== REVOLTS & EVENTS =====
  'jothiya sitana': ['jothya sitana', 'jotiya sitana', 'jothiya seetana'],
  
  // ===== FOREIGN NAMES IN CONTEXT =====
  'cheng ho': ['cheng-ho', 'zheng he', 'chengho'],
  'vasco de gama': ['vasco degama', 'vasco da gama', 'vasco the gama'],
  'constantinu de sa': ['constantino de sa', 'constantine de sa', 'constantinu'],
  
  // ===== NUMBERS (Cardinal) - commonly misrecognized =====
  'one': ['won', 'wan', '1'],
  'two': ['too', 'to', 'tu', '2'],
  'three': ['tree', 'free', '3'],
  'four': ['for', 'fore', '4'],
  'five': ['fife', 'fiv', '5'],
  'six': ['sicks', 'sic', 'sex', 'secs', '6'],
  'seven': ['sven', 'sevan', '7'],
  'eight': ['ate', 'ait', '8'],
  'nine': ['nein', 'nyne', '9'],
  'ten': ['tan', 'tin', '10'],
  
  // ===== NUMBERS (Ordinal) - commonly misrecognized =====
  'first': ['furst', 'fist', '1st'],
  'second': ['secund', 'seconded', 'sekond', '2nd'],
  'third': ['thurd', 'turd', '3rd'],
  'fourth': ['forth', 'fouth', '4th'],
  'fifth': ['fith', 'fifthe', '5th'],
  'sixth': ['sixt', 'sikth', '6th'],
  'seventh': ['sevnth', 'sevanth', '7th'],
  'eighth': ['eigth', 'eith', '8th'],
  'ninth': ['nineth', 'nynth', '9th'],
  'tenth': ['tinth', 'tanth', '10th'],
};

/**
 * Correct transcribed text by replacing misrecognized Sinhala words
 */
function correctSinhalaTranscript(text: string): string {
  let corrected = text.toLowerCase();
  
  for (const [correct, variants] of Object.entries(SINHALA_VOCABULARY)) {
    for (const variant of variants) {
      // Create a case-insensitive regex that matches whole words
      const regex = new RegExp(`\\b${variant.replace(/\s+/g, '\\s*')}\\b`, 'gi');
      corrected = corrected.replace(regex, correct);
    }
  }
  
  // Capitalize proper nouns (first letter of each word for place names)
  corrected = corrected.replace(/\b([a-z])/g, (match) => match.toUpperCase());
  
  return corrected;
}

interface VoiceRecorderProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (text: string) => void;
  title?: string;
  context?: string;
}

// Google Cloud STT API endpoint
const STT_API_URL = (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_VIS_BACKEND_URL) || 'http://localhost:5000';

/**
 * Transcribe audio using Google Cloud Speech-to-Text API with phrase hints
 */
async function transcribeWithGoogleCloud(audioBlob: Blob): Promise<string> {
  // Convert blob to base64
  const arrayBuffer = await audioBlob.arrayBuffer();
  const base64Audio = btoa(
    new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), '')
  );

  const response = await fetch(`${STT_API_URL}/api/stt/transcribe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      audio: base64Audio,
      encoding: 'WEBM_OPUS',
      sampleRateHertz: 48000,
      languageCode: 'en-LK',
    }),
  });

  if (!response.ok) {
    throw new Error(`STT API error: ${response.status}`);
  }

  const data = await response.json();
  return data.transcript || '';
}

export const VoiceRecorder: React.FC<VoiceRecorderProps> = ({ isOpen, onClose, onSubmit, title = 'Record Your Answer', context }) => {
  const [transcript, setTranscript] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [useGoogleCloud, setUseGoogleCloud] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recognitionRef = useRef<any>(null);

  // Google Cloud recording using MediaRecorder
  const startGoogleCloudRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });
      
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
        
        // Process audio
        setIsProcessing(true);
        try {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm;codecs=opus' });
          const transcribedText = await transcribeWithGoogleCloud(audioBlob);
          
          if (transcribedText) {
            // Apply local corrections as additional cleanup
            const corrected = correctSinhalaTranscript(transcribedText);
            setTranscript(prev => (prev ? prev + ' ' + corrected : corrected).trim());
          } else {
            setError('No speech detected. Please try again.');
          }
        } catch (err) {
          console.error('Google Cloud STT failed:', err);
          setError('Cloud transcription failed. Try the browser fallback.');
        } finally {
          setIsProcessing(false);
        }
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recording:', err);
      setError('Microphone access denied. Please allow microphone access.');
    }
  };

  // Web Speech API fallback
  const startBrowserRecording = () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      setError('Speech Recognition not supported in this browser.');
      return;
    }
    
    setError(null);
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-LK';
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.maxAlternatives = 3;
    
    recognition.onresult = (event: any) => {
      let finalTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          let bestText = result[0].transcript;
          let bestConfidence = result[0].confidence || 0;
          
          for (let j = 0; j < result.length; j++) {
            const alt = result[j].transcript;
            const corrected = correctSinhalaTranscript(alt);
            const hasSinhalaWords = Object.keys(SINHALA_VOCABULARY).some(word => 
              corrected.toLowerCase().includes(word)
            );
            if (hasSinhalaWords || result[j].confidence > bestConfidence) {
              bestText = alt;
              bestConfidence = result[j].confidence || bestConfidence;
            }
          }
          
          finalTranscript += correctSinhalaTranscript(bestText) + ' ';
        }
      }
      
      if (finalTranscript) {
        setTranscript(prev => (prev + ' ' + finalTranscript).trim());
      }
    };
    
    recognition.onend = () => setIsRecording(false);
    recognition.onerror = (event: any) => {
      setError('Recognition error: ' + event.error);
      setIsRecording(false);
    };
    
    recognitionRef.current = recognition;
    recognition.start();
    setIsRecording(true);
  };

  const startRecording = () => {
    if (useGoogleCloud) {
      startGoogleCloudRecording();
    } else {
      startBrowserRecording();
    }
  };

  const stopRecording = () => {
    if (useGoogleCloud && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    } else if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSubmit = () => {
    if (transcript.trim()) {
      onSubmit(transcript);
      setTranscript('');
      setError(null);
      onClose();
    }
  };

  const handleClear = () => {
    setTranscript('');
    setError(null);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
        <h2 className="text-lg font-bold mb-2">{title}</h2>
        {context && <p className="mb-2 text-sm text-gray-600">{context}</p>}
        
        {/* Mode toggle */}
        <div className="mb-3 flex items-center gap-2 text-sm">
          <label className="flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={useGoogleCloud}
              onChange={(e) => setUseGoogleCloud(e.target.checked)}
              disabled={isRecording || isProcessing}
              className="rounded"
            />
            <span className={useGoogleCloud ? 'text-blue-600 font-medium' : 'text-gray-600'}>
              Enhanced Recognition (Sinhala names)
            </span>
          </label>
        </div>
        
        {/* Error display */}
        {error && (
          <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
            {error}
          </div>
        )}
        
        {/* Processing indicator */}
        {isProcessing && (
          <div className="mb-3 p-2 bg-blue-50 border border-blue-200 rounded text-blue-700 text-sm flex items-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Processing your speech with enhanced recognition...
          </div>
        )}
        
        <div className="mb-4">
          <textarea
            className="w-full border rounded p-2 min-h-[100px]"
            value={transcript}
            onChange={e => setTranscript(e.target.value)}
            placeholder={isRecording ? 'Listening...' : 'Your answer will appear here...'}
            disabled={isRecording || isProcessing}
          />
        </div>
        
        <div className="flex flex-wrap gap-2">
          {!isRecording ? (
            <button 
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50" 
              onClick={startRecording}
              disabled={isProcessing}
            >
              🎤 Start Recording
            </button>
          ) : (
            <button 
              className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 animate-pulse" 
              onClick={stopRecording}
            >
              ⏹ Stop Recording
            </button>
          )}
          
          <button 
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50" 
            onClick={handleSubmit} 
            disabled={!transcript.trim() || isRecording || isProcessing}
          >
            ✓ Submit
          </button>
          
          {transcript && (
            <button 
              className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600" 
              onClick={handleClear}
              disabled={isRecording || isProcessing}
            >
              Clear
            </button>
          )}
          
          <button 
            className="px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500" 
            onClick={onClose}
            disabled={isRecording || isProcessing}
          >
            Cancel
          </button>
        </div>
        
        {/* Hint for users */}
        <p className="mt-3 text-xs text-gray-500">
          {useGoogleCloud 
            ? 'Using enhanced recognition for Sri Lankan historical terms (Anuradhapura, Sigiriya, Mahavamsa, etc.)'
            : 'Using browser speech recognition with local corrections'
          }
        </p>
      </div>
    </div>
  );
};
