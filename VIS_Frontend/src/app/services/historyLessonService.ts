/**
 * History Lesson service – backend for chapters, topics, and audio.
 * Uses VITE_API_URL_HISTORY. Do not mix with api.ts (document/braille/quiz).
 */

const env = (import.meta as any).env;

const VITE_API_URL_HISTORY =
  env?.VITE_API_URL_HISTORY || 'http://localhost:8003';

/** Base URL for the history/lessons backend (chapters, topics, audio). */
export const API_BASE_URL = VITE_API_URL_HISTORY;

/** Build URL for chapters by grade: GET /api/chapters/{grade} */
export function getChaptersUrl(grade: number): string {
  return `${API_BASE_URL}/api/chapters/${grade}`;
}

/** Build URL for topics: GET /api/chapters/{grade}/{chapterId}/topics */
export function getTopicsUrl(grade: number, chapterId: number): string {
  return `${API_BASE_URL}/api/chapters/${grade}/${chapterId}/topics`;
}

/** Build URL for lesson audio: GET /api/audio/chapter/{grade}/{chapterIdx}/{topicIdx} */
export function getAudioUrl(
  grade: number,
  chapterIdx: number,
  topicIdx: number
): string {
  return `${API_BASE_URL}/api/audio/chapter/${grade}/${chapterIdx}/${topicIdx}`;
}
