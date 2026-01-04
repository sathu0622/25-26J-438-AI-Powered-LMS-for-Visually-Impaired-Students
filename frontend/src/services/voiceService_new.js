// Voice Recognition and Speech Synthesis Service
class VoiceService {
  constructor() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognition = SpeechRecognition ? new SpeechRecognition() : null;
    this.synth = window.speechSynthesis;
    this.isListening = false;
    this.commandCallbacks = {};
    this.transcriptCallbacks = [];

    if (this.recognition) {
      this.setupRecognition();
    }
  }

  setupRecognition() {
    this.recognition.continuous = false;
    this.recognition.interimResults = false;
    this.recognition.language = 'en-US';

    this.recognition.onstart = () => {
      this.isListening = true;
      console.log('Voice recognition started');
    };

    this.recognition.onend = () => {
      this.isListening = false;
      console.log('Voice recognition ended');
      // Auto-restart after a short delay
      setTimeout(() => {
        if (this.recognition) {
          try {
            this.recognition.start();
            this.isListening = true;
            console.log('Voice recognition restarted');
          } catch (e) {
            console.log('Could not restart recognition:', e.message);
          }
        }
      }, 100);
    };

    this.recognition.onerror = (event) => {
      console.error('Speech recognition error', event.error);
      this.isListening = false;
      // Try to restart on error
      setTimeout(() => {
        if (this.recognition) {
          try {
            this.recognition.start();
            this.isListening = true;
          } catch (e) {
            console.log('Could not restart after error:', e.message);
          }
        }
      }, 1000);
    };

    this.recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map(result => result[0].transcript)
        .join('');

      console.log('Recognized:', transcript);
      
      // Emit transcript to all listeners
      this.transcriptCallbacks.forEach(callback => callback(transcript));
      
      this.processCommand(transcript.toLowerCase());
    };
  }

  startListening() {
    if (this.recognition && !this.isListening) {
      try {
        this.recognition.start();
        this.isListening = true;
        console.log('Voice listening started');
      } catch (e) {
        console.log('Could not start listening:', e.message);
      }
    }
  }

  stopListening() {
    if (this.recognition && this.isListening) {
      try {
        this.recognition.stop();
        this.isListening = false;
        console.log('Voice listening stopped');
      } catch (e) {
        console.log('Could not stop listening:', e.message);
      }
    }
  }

  processCommand(transcript) {
    // Command processing with more variations
    const commands = {
      'play': 'play',
      'pause': 'pause',
      'stop': 'pause',
      'next': 'next',
      'forward': 'next',
      'previous': 'previous',
      'back': 'back',
      'go back': 'back',
      'return': 'back',
      'home': 'home',
      'grade ten': 'selectGrade',
      'grade 10': 'selectGrade',
      'grade10': 'selectGrade',
      'ten': 'selectGrade',
      '10': 'selectGrade',
      'grade eleven': 'selectGrade',
      'grade 11': 'selectGrade',
      'grade11': 'selectGrade',
      'eleven': 'selectGrade',
      '11': 'selectGrade',
    };

    // Check for exact or partial matches
    for (const [keyword, command] of Object.entries(commands)) {
      if (transcript.includes(keyword)) {
        console.log(`Matched command: "${keyword}" -> ${command}`);
        this.triggerCommand(command, transcript);
        return;
      }
    }
    
    console.log('No command matched for:', transcript);
  }

  triggerCommand(command, transcript) {
    if (this.commandCallbacks[command]) {
      this.commandCallbacks[command](transcript);
    }
  }

  onCommand(command, callback) {
    this.commandCallbacks[command] = callback;
  }

  onTranscript(callback) {
    this.transcriptCallbacks.push(callback);
  }

  speak(text) {
    if (!this.synth) {
      console.log('Speech synthesis not supported');
      return;
    }

    // Cancel any ongoing speech
    this.synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;

    this.synth.speak(utterance);
  }

  async generateVoiceAudio(text) {
    // Use Web Speech API for real-time synthesis
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.onend = () => resolve(true);
      this.synth.speak(utterance);
    });
  }

  isSupported() {
    return this.recognition !== null && this.synth !== null;
  }
}

export const voiceService = new VoiceService();
export default voiceService;
