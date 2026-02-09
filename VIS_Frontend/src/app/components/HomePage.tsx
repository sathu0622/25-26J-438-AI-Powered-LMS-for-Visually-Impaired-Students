import { FileText, Image, HelpCircle, BookOpen, Volume2 } from 'lucide-react';
import { Card } from './ui/card';

interface HomePageProps {
  onNavigate: (module: string) => void;
}

const modules = [
  {
    id: 'document',
    title: 'Document AI',
    description: 'Upload PDFs or images to hear and read summaries',
    icon: FileText,
    color: 'bg-blue-500',
  },
  {
    id: 'braille',
    title: 'Braille Evaluation',
    description: 'Upload Braille answer sheets for AI evaluation',
    icon: Image,
    color: 'bg-purple-500',
  },
  {
    id: 'quiz',
    title: 'Voice Quiz',
    description: 'Interactive AI-powered quiz with voice support',
    icon: HelpCircle,
    color: 'bg-green-500',
  },
  {
    id: 'history',
    title: 'History Lessons',
    description: 'AI audio lessons for Grade 10 & 11 students',
    icon: BookOpen,
    color: 'bg-orange-500',
  },
];

export const HomePage = ({ onNavigate }: HomePageProps) => {
  return (
    <div className="mx-auto max-w-4xl space-y-8 p-4 pb-24">
      {/* Header */}
      <div className="space-y-4 text-center">
        <div className="flex justify-center">
          <div className="rounded-full bg-primary p-6">
            <Volume2 className="h-12 w-12 text-primary-foreground" aria-hidden="true" />
          </div>
        </div>
        <h1 className="text-3xl">EduVoice AI</h1>
        <p className="text-muted-foreground">
          Voice-First Educational Platform for Visually Impaired Students
        </p>
        <p className="text-sm text-secondary">
          Press <kbd className="px-2 py-1 ">F1</kbd> for voice commands or{' '}
          <kbd className="px-2 py-1  rounded">H</kbd> for help
        </p>
      </div>

      {/* Module Cards */}
      <div className="grid gap-4 sm:grid-cols-2" role="list" aria-label="Available modules">
        {modules.map((module) => {
          const Icon = module.icon;
          return (
            <Card
              key={module.id}
              className="overflow-hidden transition-all hover:shadow-lg"
              role="listitem"
            >
              <button
                onClick={() => onNavigate(module.id)}
                className="w-full p-6 text-left focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
                aria-label={`Open ${module.title} - ${module.description}`}
              >
                <div className="space-y-4">
                  <div className={`inline-flex rounded-lg ${module.color} p-3`}>
                    <Icon className="h-8 w-8 text-white" aria-hidden="true" />
                  </div>
                  <div className="space-y-2">
                    <h2 className="text-xl">{module.title}</h2>
                    <p className="text-sm text-muted-foreground">
                      {module.description}
                    </p>
                  </div>
                  <div className="flex justify-end">
                    <span
                      className="text-sm text-muted-foreground"
                      aria-hidden="true"
                    >
                      Open →
                    </span>
                  </div>
                </div>
              </button>
            </Card>
          );
        })}
      </div>

      {/* Accessibility Notice */}
      <Card className="border-secondary bg-secondary/10 p-6">
        <div className="space-y-3 text-center">
          <h2 className="text-lg">
            ♿ Accessibility Features
          </h2>
          <ul className="space-y-1 text-sm text-muted-foreground">
            <li>🎤 Voice commands - Press F1</li>
            <li>⌨️ Keyboard shortcuts - Alt+1, Alt+2, Alt+3, Alt+4</li>
            <li>🔊 Audio feedback for all actions</li>
            <li>📱 Screen reader optimized</li>
            <li>🎨 High contrast design</li>
            <li>👆 Large touch targets (56px+)</li>
          </ul>
          <div className="pt-3 border-t border-secondary mt-3">
            <p className="text-xs text-muted-foreground">
              💡 <strong>Voice Commands:</strong> Click "Allow" when browser asks for microphone permission to enable voice navigation
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};