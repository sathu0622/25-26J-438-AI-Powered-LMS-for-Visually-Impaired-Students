import { Home, FileText, Image, HelpCircle, BookOpen } from 'lucide-react';
import { Card } from './ui/card';
import { cn } from './ui/utils';

interface NavigationProps {
  currentModule: string;
  onNavigate: (module: string) => void;
}

const navItems = [
  { id: 'home', label: 'Home', icon: Home, shortcut: 'Alt+0' },
  { id: 'document', label: 'Documents', icon: FileText, shortcut: 'Alt+1' },
  { id: 'braille', label: 'Braille', icon: Image, shortcut: 'Alt+2' },
  { id: 'quiz', label: 'Quiz', icon: HelpCircle, shortcut: 'Alt+3' },
  { id: 'history', label: 'History', icon: BookOpen, shortcut: 'Alt+4' },
];

export const Navigation = ({ currentModule, onNavigate }: NavigationProps) => {
  return (
    <Card className="fixed bottom-0 left-0 right-0 z-50 rounded-none border-t bg-card shadow-lg">
      <nav
        className="flex justify-around py-2"
        role="navigation"
        aria-label="Main navigation"
      >
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentModule === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={cn(
                'flex min-h-[56px] min-w-[56px] flex-col items-center justify-center gap-1 rounded-lg px-4 transition-all focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
              aria-label={`${item.label} - Keyboard shortcut: ${item.shortcut}`}
              aria-current={isActive ? 'page' : undefined}
              title={`${item.label} (${item.shortcut})`}
            >
              <Icon className="h-6 w-6" aria-hidden="true" />
              <span className="text-xs">{item.label}</span>
            </button>
          );
        })}
      </nav>
    </Card>
  );
};