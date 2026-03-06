import React, { useEffect, useState } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import { Calendar, Trophy, Target, TrendingUp, Clock, ArrowLeft, BookOpen } from 'lucide-react';
import { useTTS } from '../contexts/TTSContext';
import { userService, UserProfile, QuizHistory } from '../services/userService';

interface UserProfilePageProps {
  username: string;
  onBack: () => void;
}

export const UserProfilePage = ({ username, onBack }: UserProfilePageProps) => {
  const { speak, cancel } = useTTS();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState(0);
  const [focusedCard, setFocusedCard] = useState<number>(-1); // For overview cards navigation

  const tabs = ['overview', 'generative', 'adaptive', 'pastpaper', 'recent'];

  useEffect(() => {
    loadUserProfile();
  }, [username]);

  useEffect(() => {
    // Announce keyboard navigation instructions on page load
    const instructions = `
      User Profile Page loaded. 
      Navigation instructions:
      Press Tab to move through elements.
      Use Left and Right arrow keys to navigate between tabs.
      Use Up and Down arrow keys to navigate overview cards.
      Press Enter to activate buttons.
      Press Escape or Backspace to go back.
      Press H to hear these instructions again.
    `;
    speak(instructions, { interrupt: false });

    // Cleanup function to cancel speech when component unmounts
    return () => {
      cancel();
    };
  }, [speak, cancel]);

  useEffect(() => {
    // Enhanced keyboard navigation
    const handleKeys = (e: KeyboardEvent) => {
      // Help instructions
      if (e.key === 'h' || e.key === 'H') {
        e.preventDefault();
        const instructions = `
          Navigation Help:
          Tab: Move through elements
          Left/Right arrows: Switch between tabs
          Up/Down arrows: Navigate overview cards
          Enter: Activate buttons
          Escape/Backspace: Go back
          H: Repeat instructions
        `;
        speak(instructions, { interrupt: true });
        return;
      }

      if (e.key === 'Backspace' || e.key === 'Escape') {
        cancel(); // Stop any ongoing speech
        speak('Returning to quiz mode selection', { interrupt: true });
        // Small delay to let the announcement play before navigation
        setTimeout(() => {
          onBack();
        }, 1000);
        return;
      }

      if (e.key === 'ArrowLeft' && !e.shiftKey) {
        e.preventDefault();
        const newTab = (selectedTab - 1 + tabs.length) % tabs.length;
        setSelectedTab(newTab);
        setFocusedCard(-1);
        return;
      }

      if (e.key === 'ArrowRight' && !e.shiftKey) {
        e.preventDefault();
        const newTab = (selectedTab + 1) % tabs.length;
        setSelectedTab(newTab);
        setFocusedCard(-1);
        return;
      }

      // Overview cards navigation (only in overview tab)
      if (selectedTab === 0) {
        if (e.key === 'ArrowUp') {
          e.preventDefault();
          const newCard = focusedCard <= 0 ? 4 : focusedCard - 1;
          setFocusedCard(newCard);
          announceOverviewCard(newCard);
        }

        if (e.key === 'ArrowDown') {
          e.preventDefault();
          const newCard = focusedCard >= 4 ? 0 : focusedCard + 1;
          setFocusedCard(newCard);
          announceOverviewCard(newCard);
        }
      }
    };

    document.addEventListener('keydown', handleKeys);
    return () => {
      document.removeEventListener('keydown', handleKeys);
      // Cancel any ongoing speech when keyboard listeners are removed
      cancel();
    };
  }, [tabs.length, onBack, selectedTab, focusedCard, speak, cancel]);

  const announceOverviewCard = (cardIndex: number) => {
    if (!profile) return;
    cancel(); // Cancel any previous announcement
    const cards = [
      `Total Quizzes card selected: You have completed ${profile.total_quizzes} ${profile.total_quizzes === 1 ? 'quiz' : 'quizzes'} in total. This includes generative, adaptive, and past paper quiz types.`,
      `Average Score card selected: Your overall performance average is ${profile.average_score} percent. ${profile.average_score >= 80 ? 'Excellent work! You are performing very well.' : profile.average_score >= 60 ? 'Good progress! Keep practicing to improve further.' : 'Keep learning! Your scores will improve with more practice.'}`,
      `Generative Quizzes card selected: You have completed ${profile.generative_quizzes} generative ${profile.generative_quizzes === 1 ? 'quiz' : 'quizzes'}. Generative quizzes use AI to create questions from your selected chapters.`,
      `Adaptive Quizzes card selected: You have completed ${profile.adaptive_quizzes} adaptive ${profile.adaptive_quizzes === 1 ? 'quiz' : 'quizzes'}. Adaptive quizzes automatically adjust difficulty based on your performance to provide personalized learning.`,
      `Past Paper Quizzes card selected: You have completed ${profile.past_paper_quizzes} past paper ${profile.past_paper_quizzes === 1 ? 'quiz' : 'quizzes'}. Past paper quizzes test you on real exam questions from previous years organized by chapter.`
    ];
    speak(cards[cardIndex], { interrupt: true });
  };

  useEffect(() => {
    if (profile && !loading) {
      // Cancel previous speech before announcing new tab
      cancel();
      
      const tabName = tabs[selectedTab];
      let announcement = '';
      
      switch(selectedTab) {
        case 0: // Overview
          announcement = `
            Overview tab selected. Performance Summary:
            Your overall average score is ${profile.average_score} percent.
            You have completed ${profile.total_quizzes} total quizzes.
            Quiz distribution: ${profile.generative_quizzes} generative quizzes, ${profile.adaptive_quizzes} adaptive quizzes, and ${profile.past_paper_quizzes} past paper quizzes.
            ${profile.recent_activity.length > 0 
              ? `Your last quiz was completed on ${formatDate(profile.recent_activity[0].completed_at)} with a score of ${profile.recent_activity[0].score} percent.`
              : 'No recent quiz activity found.'
            }
            Use up and down arrows to navigate overview cards for detailed statistics.
          `;
          break;
        case 1: // Generative
          const genCount = profile.quiz_history.generative.length;
          announcement = `
            Generative Quizzes tab selected. 
            You have completed ${genCount} generative ${genCount === 1 ? 'quiz' : 'quizzes'}.
            ${genCount > 0 
              ? `Your average score in generative quizzes is ${Math.round(profile.quiz_history.generative.reduce((sum, q) => sum + q.score, 0) / genCount)} percent. Use Tab to navigate through quiz details.`
              : 'No generative quizzes completed yet. Start your first generative quiz to see your history here.'
            }
          `;
          break;
        case 2: // Adaptive
          const adaptCount = profile.quiz_history.adaptive.length;
          announcement = `
            Adaptive Quizzes tab selected.
            You have completed ${adaptCount} adaptive ${adaptCount === 1 ? 'quiz' : 'quizzes'}.
            ${adaptCount > 0 
              ? `Your average score in adaptive quizzes is ${Math.round(profile.quiz_history.adaptive.reduce((sum, q) => sum + q.score, 0) / adaptCount)} percent. Adaptive quizzes adjust difficulty based on your performance. Use Tab to navigate through quiz details.`
              : 'No adaptive quizzes completed yet. Try your first adaptive quiz to see personalized difficulty adjustment.'
            }
          `;
          break;
        case 3: // Past Paper
          const pastPaperCount = profile.quiz_history.past_paper?.length || 0;
          const chapterStats = getPastPaperChapterStats();
          announcement = `
            Past Paper Quizzes tab selected.
            You have completed ${pastPaperCount} past paper ${pastPaperCount === 1 ? 'quiz' : 'quizzes'}.
            ${pastPaperCount > 0 
              ? `Performance is organized by ${chapterStats.length} ${chapterStats.length === 1 ? 'chapter' : 'chapters'}. ${chapterStats.length > 0 ? `Best performing chapter: ${chapterStats[0].chapter} with ${chapterStats[0].averageScore} percent average.` : ''} Use Tab to navigate through chapter performance details.`
              : 'No past paper quizzes completed yet. Try a past paper quiz to test yourself on real exam questions.'
            }
          `;
          break;
        case 4: // Recent
          const recentCount = profile.recent_activity.length;
          announcement = `
            Recent Activity tab selected.
            ${recentCount > 0 
              ? `Showing your ${recentCount} most recent quiz ${recentCount === 1 ? 'attempt' : 'attempts'}. Your latest activity was ${formatDate(profile.recent_activity[0].completed_at)} scoring ${profile.recent_activity[0].score} percent. Use Tab to navigate through recent quiz details.`
              : 'No recent quiz activity found. Complete some quizzes to see your recent performance here.'
            }
          `;
          break;
      }
      
      speak(announcement, { interrupt: true });
    }

    // Cleanup function to cancel speech when tab changes
    return () => {
      cancel();
    };
  }, [selectedTab, profile, loading, tabs, speak, cancel]);

  // Helper function to get past paper stats organized by chapter
  const getPastPaperChapterStats = () => {
    if (!profile || !profile.quiz_history.past_paper) return [];
    
    const chapterMap: { [chapter: string]: { total: number; score: number; count: number; quizzes: QuizHistory[] } } = {};
    
    profile.quiz_history.past_paper.forEach(quiz => {
      const chapter = quiz.chapter_name;
      if (!chapterMap[chapter]) {
        chapterMap[chapter] = { total: 0, score: 0, count: 0, quizzes: [] };
      }
      chapterMap[chapter].total += quiz.total_questions;
      chapterMap[chapter].score += quiz.score;
      chapterMap[chapter].count += 1;
      chapterMap[chapter].quizzes.push(quiz);
    });
    
    return Object.entries(chapterMap)
      .map(([chapter, data]) => ({
        chapter,
        totalQuizzes: data.count,
        totalQuestions: data.total,
        averageScore: Math.round(data.score / data.count),
        quizzes: data.quizzes.sort((a, b) => new Date(b.completed_at).getTime() - new Date(a.completed_at).getTime())
      }))
      .sort((a, b) => b.averageScore - a.averageScore);
  };

  const handleBackNavigation = () => {
    // Cancel any ongoing speech before navigating back
    cancel();
    speak('Leaving profile page', { interrupt: true });
    // Small delay to let the announcement play before navigation
    setTimeout(() => {
      onBack();
    }, 500);
  };

  const loadUserProfile = async () => {
    try {
      setLoading(true);
      cancel(); // Cancel any ongoing speech before loading
      speak('Loading your profile data', { interrupt: true });
      const data = await userService.getUserProfile(username);
      setProfile(data);
      
      // Detailed profile announcement
      const announcement = `
        Profile loaded successfully for ${data.username}. 
        Statistics overview: 
        ${data.total_quizzes} total quizzes completed with an average score of ${data.average_score} percent.
        ${data.generative_quizzes} generative quizzes and ${data.adaptive_quizzes} adaptive quizzes completed.
        Use keyboard navigation to explore your detailed performance data.
      `; 
      speak(announcement, { interrupt: true });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to load profile data. Please try again.';
      setError(errorMsg);
      speak(`Error: ${errorMsg}`, { interrupt: true });
    } finally {
      setLoading(false);
    }
  };

  const getTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 3600 * 24));
    
    if (diffInDays === 0) return 'today';
    if (diffInDays === 1) return 'yesterday';
    if (diffInDays < 7) return `${diffInDays} days ago`;
    if (diffInDays < 30) return `${Math.floor(diffInDays / 7)} week${Math.floor(diffInDays / 7) === 1 ? '' : 's'} ago`;
    if (diffInDays < 365) return `${Math.floor(diffInDays / 30)} month${Math.floor(diffInDays / 30) === 1 ? '' : 's'} ago`;
    return `${Math.floor(diffInDays / 365)} year${Math.floor(diffInDays / 365) === 1 ? '' : 's'} ago`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBadgeVariant = (score: number) => {
    if (score >= 80) return 'default';
    if (score >= 60) return 'secondary';
    return 'destructive';
  };

  if (loading) {
    return (
      <div className="p-6 text-center" role="status" aria-live="polite">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4" aria-hidden="true"></div>
        <p>Loading your profile...</p>
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="p-6 text-center" role="alert" aria-live="assertive">
        <p className="text-red-600 mb-4">{error || 'Profile not found'}</p>
        <Button onClick={handleBackNavigation} variant="outline" aria-label="Return to quiz selection">
          <ArrowLeft className="mr-2 h-4 w-4" aria-hidden="true" />
          Back
        </Button>
      </div>
    );
  }

  const QuizHistoryTable = ({ quizzes, type }: { quizzes: QuizHistory[], type: string }) => (
    <section className="space-y-4" role="region" aria-labelledby={`${type}-history-heading`}>
      <h4 id={`${type}-history-heading`} className="sr-only">{type} Quiz History List</h4>
      {quizzes.length === 0 ? (
        <div className="text-muted-foreground text-center py-8" role="status" aria-live="polite">
          No {type} quizzes completed yet.
        </div>
      ) : (
        <div role="list" aria-label={`${quizzes.length} ${type} quiz${quizzes.length !== 1 ? 'es' : ''} completed`}>
          {quizzes.map((quiz, index) => (
            <Card 
              key={index} 
              className="p-4" 
              role="listitem"
              aria-labelledby={`quiz-${type}-${index}-title`}
              tabIndex={0}
              onFocus={() => {
                cancel(); // Cancel any previous announcement
                const scoreDescription = quiz.score >= 80 ? 'excellent performance' : quiz.score >= 60 ? 'good performance' : 'needs improvement';
                const timeAgo = getTimeAgo(quiz.completed_at);
                
                let announcement = `
                  ${quiz.chapter_name} ${type} quiz details:
                  Completed ${timeAgo} on ${formatDate(quiz.completed_at)}.
                  Your score: ${quiz.score} percent - ${scoreDescription}.
                  Total questions: ${quiz.total_questions}.
                `;
                
                if (quiz.quiz_type === 'Generative' && quiz.correct_answers !== undefined) {
                  announcement += `
                    Correct answers: ${quiz.correct_answers} out of ${quiz.total_questions}.
                    Accuracy rate: ${Math.round((quiz.correct_answers / quiz.total_questions) * 100)} percent.
                  `;
                }
                
                if (quiz.quiz_type === 'Adaptive' && quiz.theta !== undefined) {
                  announcement += `
                    Final skill level reached: ${quiz.final_level}.
                    Ability score: ${quiz.theta?.toFixed(2)} points.
                    The adaptive system adjusted question difficulty based on your responses.
                  `;
                }
                
                speak(announcement, { interrupt: true });
              }}
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <h5 
                    id={`quiz-${type}-${index}-title`} 
                    className="font-medium text-lg"
                  >
                    {quiz.chapter_name}
                  </h5>
                  <p className="text-sm text-muted-foreground">
                    <time dateTime={quiz.completed_at}>
                      {formatDate(quiz.completed_at)}
                    </time>
                  </p>
                </div>
                <div className="text-right">
                  <Badge 
                    variant={getScoreBadgeVariant(quiz.score)} 
                    className="mb-2"
                    aria-label={`Score: ${quiz.score} percent`}
                  >
                    {quiz.score}%
                  </Badge>
                  <p className="text-sm text-muted-foreground">
                    {quiz.total_questions} question{quiz.total_questions !== 1 ? 's' : ''}
                  </p>
                </div>
              </div>
              
              {quiz.quiz_type === 'Generative' && quiz.correct_answers !== undefined && (
                <div 
                  className="text-sm text-muted-foreground"
                  aria-label={`Correct answers: ${quiz.correct_answers} out of ${quiz.total_questions}`}
                >
                  Correct: {quiz.correct_answers}/{quiz.total_questions}
                </div>
              )}
              
              {quiz.quiz_type === 'Adaptive' && quiz.theta !== undefined && (
                <div className="text-sm text-muted-foreground space-y-1" role="group" aria-label="Adaptive quiz metrics">
                  <div>Skill Level: {quiz.final_level}</div>
                  <div>Ability Score: {quiz.theta?.toFixed(2)}</div>
                </div>
              )}
            </Card>
          ))}
        </div>
      )}
    </section>
  );

  return (
    <main className="p-6 max-w-6xl mx-auto" role="main" aria-label="User Profile Page">
      {/* Skip to main content link for screen readers */}
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-primary text-white p-2 rounded">
        Skip to main content
      </a>
      
      {/* Header */}
      <header className="flex items-center justify-between mb-6" role="banner">
        <div className="flex items-center space-x-4">
          <Button 
            onClick={handleBackNavigation} 
            variant="outline" 
            size="sm"
            aria-label="Return to quiz mode selection"
            tabIndex={0}
          >
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          </Button>
          <div>
            <h1 className="text-3xl font-bold" id="page-title">
              {profile.username}'s Profile
            </h1>
            <p className="text-muted-foreground">Quiz performance and history</p>
          </div>
        </div>
      </header>

      {/* Navigation Help Text */}
      <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg" role="complementary" aria-labelledby="nav-help">
        <h2 id="nav-help" className="text-sm font-semibold mb-1">Keyboard Navigation</h2>
        <p className="text-xs text-muted-foreground">
          Use Left/Right arrows for tabs, Up/Down arrows for overview cards, H for help, Tab to move through elements
        </p>
      </div>

      {/* Profile Overview Cards */}
      <section 
        className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6" 
        id="main-content"
        role="region" 
        aria-labelledby="overview-heading"
      >
        <h2 id="overview-heading" className="sr-only">Performance Overview Cards</h2>
        
        <Card 
          className={`p-6 text-center ${focusedCard === 0 ? 'ring-2 ring-blue-500' : ''}`}
          role="region"
          aria-labelledby="total-quizzes-heading"
          tabIndex={0}
          onFocus={() => {
            cancel(); // Cancel any previous speech
            speak(`Total Quizzes card focused. You have completed ${profile.total_quizzes} ${profile.total_quizzes === 1 ? 'quiz' : 'quizzes'} overall. This includes both generative AI quizzes and adaptive difficulty quizzes. Great job on your learning journey!`, { interrupt: true });
          }}
        >
          <Trophy className="h-8 w-8 text-yellow-500 mx-auto mb-2" aria-hidden="true" />
          <div className="text-2xl font-bold" aria-label={`${profile.total_quizzes} quizzes completed`}>
            {profile.total_quizzes}
          </div>
          <div id="total-quizzes-heading" className="text-sm text-muted-foreground">Total Quizzes</div>
        </Card>
        
        <Card 
          className={`p-6 text-center ${focusedCard === 1 ? 'ring-2 ring-blue-500' : ''}`}
          role="region"
          aria-labelledby="average-score-heading"
          tabIndex={0}
          onFocus={() => {
            cancel(); // Cancel any previous speech
            const performanceLevel = profile.average_score >= 80 ? 'excellent' : profile.average_score >= 60 ? 'good' : 'developing';
            const encouragement = profile.average_score >= 80 ? 'Outstanding work! You are mastering the material.' : profile.average_score >= 60 ? 'Good progress! Continue practicing to reach excellence.' : 'Keep going! Every quiz helps you learn and improve.';
            speak(`Average Score card focused. Your overall performance average is ${profile.average_score} percent, which represents ${performanceLevel} performance. ${encouragement}`, { interrupt: true });
          }}
        >
          <Target className="h-8 w-8 text-blue-500 mx-auto mb-2" aria-hidden="true" />
          <div className="text-2xl font-bold" aria-label={`${profile.average_score} percent average score`}>
            {profile.average_score}%
          </div>
          <div id="average-score-heading" className="text-sm text-muted-foreground">Average Score</div>
        </Card>
        
        <Card 
          className={`p-6 text-center ${focusedCard === 2 ? 'ring-2 ring-blue-500' : ''}`}
          role="region"
          aria-labelledby="generative-quizzes-heading"
          tabIndex={0}
          onFocus={() => {
            cancel(); // Cancel any previous speech
            speak(`Generative Quizzes card focused. You have completed ${profile.generative_quizzes} generative ${profile.generative_quizzes === 1 ? 'quiz' : 'quizzes'}. Generative quizzes use artificial intelligence to create unique questions from your study material. ${profile.generative_quizzes > 0 ? 'Check the Generative tab for detailed results.' : 'Try a generative quiz to get AI-powered questions tailored to your learning.'}`, { interrupt: true });
          }}
        >
          <TrendingUp className="h-8 w-8 text-green-500 mx-auto mb-2" aria-hidden="true" />
          <div className="text-2xl font-bold" aria-label={`${profile.generative_quizzes} generative quizzes completed`}>
            {profile.generative_quizzes}
          </div>
          <div id="generative-quizzes-heading" className="text-sm text-muted-foreground">Generative Quizzes</div>
        </Card>
        
        <Card 
          className={`p-6 text-center ${focusedCard === 3 ? 'ring-2 ring-blue-500' : ''}`}
          role="region"
          aria-labelledby="adaptive-quizzes-heading"
          tabIndex={0}
          onFocus={() => {
            cancel(); // Cancel any previous speech
            speak(`Adaptive Quizzes card focused. You have completed ${profile.adaptive_quizzes} adaptive ${profile.adaptive_quizzes === 1 ? 'quiz' : 'quizzes'}. Adaptive quizzes automatically adjust question difficulty based on your responses to provide personalized learning. ${profile.adaptive_quizzes > 0 ? 'Check the Adaptive tab to see how the system adapted to your skill level.' : 'Try an adaptive quiz to experience personalized difficulty adjustment.'}`, { interrupt: true });
          }}
        >
          <Calendar className="h-8 w-8 text-purple-500 mx-auto mb-2" aria-hidden="true" />
          <div className="text-2xl font-bold" aria-label={`${profile.adaptive_quizzes} adaptive quizzes completed`}>
            {profile.adaptive_quizzes}
          </div>
          <div id="adaptive-quizzes-heading" className="text-sm text-muted-foreground">Adaptive Quizzes</div>
        </Card>
        
        <Card 
          className={`p-6 text-center ${focusedCard === 4 ? 'ring-2 ring-blue-500' : ''}`}
          role="region"
          aria-labelledby="past-paper-quizzes-heading"
          tabIndex={0}
          onFocus={() => {
            cancel(); // Cancel any previous speech
            speak(`Past Paper Quizzes card focused. You have completed ${profile.past_paper_quizzes} past paper ${profile.past_paper_quizzes === 1 ? 'quiz' : 'quizzes'}. Past paper quizzes test you on real exam questions from previous years. ${profile.past_paper_quizzes > 0 ? 'Check the Past Paper tab to see your performance organized by chapter.' : 'Try a past paper quiz to practice with real exam questions.'}`, { interrupt: true });
          }}
        >
          <BookOpen className="h-8 w-8 text-orange-500 mx-auto mb-2" aria-hidden="true" />
          <div className="text-2xl font-bold" aria-label={`${profile.past_paper_quizzes} past paper quizzes completed`}>
            {profile.past_paper_quizzes}
          </div>
          <div id="past-paper-quizzes-heading" className="text-sm text-muted-foreground">Past Paper Quizzes</div>
        </Card>
      </section>

      {/* Detailed Tabs */}
      <section role="region" aria-labelledby="detailed-tabs-heading">
        <h2 id="detailed-tabs-heading" className="sr-only">Detailed Quiz History and Performance</h2>
        <Tabs 
          value={tabs[selectedTab]} 
          onValueChange={(value) => setSelectedTab(tabs.indexOf(value))} 
          className="w-full"
          orientation="horizontal"
        >
          <TabsList className="grid w-full grid-cols-5" role="tablist" aria-label="Profile data categories">
            <TabsTrigger 
              value="overview" 
              role="tab" 
              aria-selected={selectedTab === 0}
              aria-controls="overview-panel"
            >
              Overview
            </TabsTrigger>
            <TabsTrigger 
              value="generative" 
              role="tab" 
              aria-selected={selectedTab === 1}
              aria-controls="generative-panel"
            >
              Generative
            </TabsTrigger>
            <TabsTrigger 
              value="adaptive" 
              role="tab" 
              aria-selected={selectedTab === 2}
              aria-controls="adaptive-panel"
            >
              Adaptive
            </TabsTrigger>
            <TabsTrigger 
              value="pastpaper" 
              role="tab" 
              aria-selected={selectedTab === 3}
              aria-controls="pastpaper-panel"
            >
              Past Paper
            </TabsTrigger>
            <TabsTrigger 
              value="recent" 
              role="tab" 
              aria-selected={selectedTab === 4}
              aria-controls="recent-panel"
            >
              Recent
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-6" role="tabpanel" id="overview-panel" aria-labelledby="overview-tab">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center" id="performance-summary-heading">
                <Trophy className="mr-2 h-5 w-5" aria-hidden="true" />
                Performance Summary
              </h3>
              <div className="space-y-4" role="group" aria-labelledby="performance-summary-heading">
                <div>
                  <div className="flex justify-between mb-2">
                    <span>Overall Average Score</span>
                    <span className={`font-medium ${getScoreColor(profile.average_score)}`}>
                      {profile.average_score}%
                    </span>
                  </div>
                  <Progress 
                    value={profile.average_score} 
                    className="w-full" 
                    aria-label={`Performance progress: ${profile.average_score} percent`}
                    role="progressbar"
                    aria-valuenow={profile.average_score}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    onFocus={() => {
                      cancel(); // Cancel any previous speech
                      const performanceLevel = profile.average_score >= 80 ? 'excellent' : profile.average_score >= 60 ? 'good' : 'needs improvement';
                      speak(`Performance progress bar focused. Your current average score is ${profile.average_score} percent which is ${performanceLevel} performance. Keep working to maintain or improve your scores.`, { interrupt: true });
                    }}
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-sm text-muted-foreground">Quiz Distribution</h4>
                  <div 
                    className="mt-2" 
                    role="list" 
                    aria-label="Quiz type distribution"
                    tabIndex={0}
                    onFocus={() => {
                      cancel(); // Cancel any previous speech
                      speak(`Quiz Distribution section. You have completed ${profile.generative_quizzes} generative quizzes and ${profile.adaptive_quizzes} adaptive quizzes. ${profile.generative_quizzes > profile.adaptive_quizzes ? 'You have done more generative quizzes.' : profile.adaptive_quizzes > profile.generative_quizzes ? 'You have done more adaptive quizzes.' : 'You have an equal split between both quiz types.'}`, { interrupt: true });
                    }}
                  >
                      <div className="flex justify-between" role="listitem">
                        <span>Generative:</span>
                        <span>{profile.generative_quizzes}</span>
                      </div>
                      <div className="flex justify-between" role="listitem">
                        <span>Adaptive:</span>
                        <span>{profile.adaptive_quizzes}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-sm text-muted-foreground">Recent Activity</h4>
                    <div 
                      className="mt-2"
                      tabIndex={0}
                      onFocus={() => {
                        cancel(); // Cancel any previous speech
                        if (profile.recent_activity.length > 0) {
                          const lastQuiz = profile.recent_activity[0];
                          const timeAgo = getTimeAgo(lastQuiz.completed_at);
                          speak(`Recent Activity section. Your most recent quiz was ${lastQuiz.chapter_name} completed ${timeAgo} with a score of ${lastQuiz.score} percent. ${lastQuiz.score >= 80 ? 'Excellent recent performance!' : lastQuiz.score >= 60 ? 'Good recent progress!' : 'Your recent attempt shows room for improvement - keep practicing!'}`, { interrupt: true });
                        } else {
                          speak('Recent Activity section. No recent quiz activity found. Complete some quizzes to see your recent performance here.', { interrupt: true });
                        }
                      }}
                    >
                      {profile.recent_activity.length > 0 ? (
                        <div className="text-sm">
                          Last quiz: {formatDate(profile.recent_activity[0].completed_at)}
                        </div>
                      ) : (
                        <div className="text-sm text-muted-foreground">No recent activity</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="generative" className="mt-6" role="tabpanel" id="generative-panel" aria-labelledby="generative-tab">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4">Generative Quiz History</h3>
              <QuizHistoryTable quizzes={profile.quiz_history.generative} type="generative" />
            </Card>
          </TabsContent>

          <TabsContent value="adaptive" className="mt-6" role="tabpanel" id="adaptive-panel" aria-labelledby="adaptive-tab">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4">Adaptive Quiz History</h3>
              <QuizHistoryTable quizzes={profile.quiz_history.adaptive} type="adaptive" />
            </Card>
          </TabsContent>

          <TabsContent value="pastpaper" className="mt-6" role="tabpanel" id="pastpaper-panel" aria-labelledby="pastpaper-tab">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                <BookOpen className="mr-2 h-5 w-5" aria-hidden="true" />
                Past Paper Quiz Performance by Chapter
              </h3>
              {getPastPaperChapterStats().length === 0 ? (
                <div className="text-muted-foreground text-center py-8" role="status" aria-live="polite">
                  No past paper quizzes completed yet. Complete a past paper quiz to see your performance here.
                </div>
              ) : (
                <div className="space-y-6" role="list" aria-label="Chapter performance list">
                  {getPastPaperChapterStats().map((chapterStat, index) => (
                    <Card 
                      key={chapterStat.chapter} 
                      className="p-4 border-l-4 border-l-orange-500"
                      role="listitem"
                      tabIndex={0}
                      onFocus={() => {
                        cancel();
                        const performanceLevel = chapterStat.averageScore >= 80 ? 'excellent' : chapterStat.averageScore >= 60 ? 'good' : 'needs improvement';
                        speak(`${chapterStat.chapter} performance summary. You have completed ${chapterStat.totalQuizzes} ${chapterStat.totalQuizzes === 1 ? 'quiz' : 'quizzes'} in this chapter. Average score: ${chapterStat.averageScore} percent, which is ${performanceLevel}. Total questions answered: ${chapterStat.totalQuestions}.`, { interrupt: true });
                      }}
                    >
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h4 className="font-semibold text-lg">{chapterStat.chapter}</h4>
                          <p className="text-sm text-muted-foreground">
                            {chapterStat.totalQuizzes} {chapterStat.totalQuizzes === 1 ? 'quiz' : 'quizzes'} completed
                          </p>
                        </div>
                        <Badge 
                          variant={getScoreBadgeVariant(chapterStat.averageScore)} 
                          className="text-lg px-3 py-1"
                          aria-label={`Average score: ${chapterStat.averageScore} percent`}
                        >
                          {chapterStat.averageScore}%
                        </Badge>
                      </div>
                      
                      <div className="mb-3">
                        <Progress value={chapterStat.averageScore} className="h-2" />
                      </div>
                      
                      <div className="text-sm text-muted-foreground mb-3">
                        Total questions: {chapterStat.totalQuestions}
                      </div>
                      
                      {/* Show individual quiz attempts for this chapter */}
                      <details className="mt-2">
                        <summary className="cursor-pointer text-sm font-medium text-blue-600 hover:text-blue-800">
                          View individual quiz attempts ({chapterStat.quizzes.length})
                        </summary>
                        <div className="mt-3 space-y-2 pl-4 border-l-2 border-gray-200">
                          {chapterStat.quizzes.map((quiz, qIndex) => (
                            <div 
                              key={qIndex} 
                              className="flex justify-between items-center py-2 text-sm"
                              tabIndex={0}
                              onFocus={() => {
                                cancel();
                                speak(`Quiz attempt ${qIndex + 1}. Completed ${formatDate(quiz.completed_at)}. Score: ${quiz.score} percent. ${quiz.correct_count || 0} correct out of ${quiz.total_questions} questions.`, { interrupt: true });
                              }}
                            >
                              <div>
                                <span className="text-muted-foreground">{formatDate(quiz.completed_at)}</span>
                                <span className="ml-2">
                                  {quiz.correct_count || 0}/{quiz.total_questions} correct
                                </span>
                              </div>
                              <Badge variant={getScoreBadgeVariant(quiz.score)} className="ml-2">
                                {quiz.score}%
                              </Badge>
                            </div>
                          ))}
                        </div>
                      </details>
                    </Card>
                  ))}
                </div>
              )}
            </Card>
          </TabsContent>

          <TabsContent value="recent" className="mt-6" role="tabpanel" id="recent-panel" aria-labelledby="recent-tab">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                <Clock className="mr-2 h-5 w-5" aria-hidden="true" />
                Recent Activity
              </h3>
              <QuizHistoryTable quizzes={profile.recent_activity} type="recent" />
            </Card>
          </TabsContent>
        </Tabs>
      </section>
    </main>
  );
};
