import { useEffect, useState } from 'react';
import { Volume2, Loader2 } from 'lucide-react';
import { adaptiveService } from '../../services/adaptiveService';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { useTTS } from '../../contexts/TTSContext';

interface AdaptiveStartProps {
  onStart: (chapter: string) => void;
}

export const AdaptiveStart = ({ onStart }: AdaptiveStartProps) => {
  const [chapters, setChapters] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const { speak, cancel } = useTTS();

  useEffect(() => {
    const load = async () => {
      try {
        const res = await adaptiveService.getChapters();
        setChapters(res);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  useEffect(() => {
    if (!loading && chapters.length) {
      speak('Select a chapter to start the adaptive quiz. Use the arrow keys to move and enter to select.');
    }
    return () => cancel();
  }, [loading, chapters, speak, cancel]);

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center gap-2 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading adaptive chapters…
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6 pb-24">
      <div className="text-center space-y-2">
        <p className="text-xs uppercase tracking-wide text-primary font-semibold">Adaptive Mode</p>
        <h1 className="text-3xl font-semibold">Choose a chapter</h1>
        <p className="text-sm text-muted-foreground">
          Optimized for screen readers: use Tab / Shift+Tab to move, Enter to select, and Space to start. Press the speaker to hear instructions again.
        </p>
        <div className="flex justify-center">
          <Button variant="ghost" size="icon" aria-label="Speak instructions" onClick={() => speak('Select a chapter to start the adaptive quiz. Use the arrow keys or tab to move and enter to select.')}>
            <Volume2 className="h-5 w-5" />
          </Button>
        </div>
      </div>

      <div className="grid gap-3" role="list" aria-label="Adaptive quiz chapters">
        {chapters.map((c, i) => {
          const isSelected = selected === c;
          return (
            <Card
              role="button"
              tabIndex={0}
              key={c}
              onClick={() => setSelected(c)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSelected(c);
                }
              }}
              aria-pressed={isSelected}
              className={`p-4 cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary/60 ${isSelected ? 'border-primary border-2 shadow-sm' : ''}`}
            >
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 rounded-full bg-muted flex items-center justify-center text-sm font-semibold" aria-hidden>
                  {i + 1}
                </div>
                <div>
                  <p className="font-medium">{c}</p>
                  <p className="text-xs text-muted-foreground">Adaptive sequencing based on your answers</p>
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      <div className="grid sm:grid-cols-2 gap-3">
        <Button
          className="w-full"
          disabled={!selected}
          onClick={() => onStart(selected)}
          aria-disabled={!selected}
        >
          Start Adaptive Quiz
        </Button>
        <Button
          variant="outline"
          className="w-full"
          onClick={() => speak(`You have ${chapters.length} chapters available. ${chapters.join(', ')}`)}
        >
          <Volume2 className="mr-2 h-4 w-4" /> Read chapters aloud
        </Button>
      </div>

      <div className="sr-only" aria-live="polite">
        {selected ? `Selected ${selected}` : 'No chapter selected'}
      </div>
    </div>
  );
};
