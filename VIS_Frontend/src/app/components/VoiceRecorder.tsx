import React, { useState, useRef } from 'react';

interface VoiceRecorderProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (text: string) => void;
  title?: string;
  context?: string;
}

export const VoiceRecorder: React.FC<VoiceRecorderProps> = ({ isOpen, onClose, onSubmit, title = 'Record Your Answer', context }) => {
  const [transcript, setTranscript] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef<any>(null);

  const startRecording = () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      alert('Speech Recognition API not supported in this browser.');
      return;
    }
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-LK';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event: any) => {
      const text = event.results[0][0].transcript;
      setTranscript(text);
    };
    recognition.onend = () => {
      setIsRecording(false);
    };
    recognition.onerror = (event: any) => {
      alert('Error occurred in recognition: ' + event.error);
      setIsRecording(false);
    };
    recognitionRef.current = recognition;
    recognition.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSubmit = () => {
    if (transcript.trim()) {
      onSubmit(transcript);
      setTranscript('');
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
        <h2 className="text-lg font-bold mb-2">{title}</h2>
        {context && <p className="mb-2 text-sm text-gray-600">{context}</p>}
        <div className="mb-4">
          <textarea
            className="w-full border rounded p-2 min-h-[80px]"
            value={transcript}
            onChange={e => setTranscript(e.target.value)}
            placeholder="Your answer will appear here..."
            disabled={isRecording}
          />
        </div>
        <div className="flex gap-2">
          {!isRecording ? (
            <button className="px-4 py-2 bg-blue-600 text-white rounded" onClick={startRecording}>
              Start Recording
            </button>
          ) : (
            <button className="px-4 py-2 bg-red-600 text-white rounded" onClick={stopRecording}>
              Stop Recording
            </button>
          )}
          <button className="px-4 py-2 bg-green-600 text-white rounded" onClick={handleSubmit} disabled={!transcript.trim()}>
            Submit
          </button>
          <button className="px-4 py-2 bg-gray-400 text-white rounded" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};
