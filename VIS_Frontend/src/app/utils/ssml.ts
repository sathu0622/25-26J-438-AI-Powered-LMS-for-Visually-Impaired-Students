/**
 * SSML (Speech Synthesis Markup Language) utilities for Google Cloud Text-to-Speech.
 * Tuning for pronunciation, speed, and emphasis for visually impaired users.
 */

/** Escape text for use inside XML/SSML */
export function escapeSsml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export interface SsmlOptions {
  /** Speaking rate: 0.25–4.0 (1.0 = normal). Slightly slower improves comprehension. */
  rate?: number;
  /** Words or phrases to emphasize (wrapped in <emphasis>). */
  emphasis?: string[];
  /** Optional prosody pitch: "x-low" | "low" | "medium" | "high" | "x-high" or "+n%" / "-n%" */
  pitch?: string;
  /** Optional break before/after (e.g. "500ms"). */
  breakAfter?: string;
  /** Wrap numbers/dates for clear pronunciation (e.g. "characters", "digits"). */
  sayAs?: 'characters' | 'digits' | 'spell-out' | 'date' | 'time';
}

const DEFAULT_RATE = 0.95; // Slightly slower for clarity (visually impaired users)

/**
 * Build Google Cloud TTS SSML from plain text with optional prosody and emphasis.
 * Uses <speak>, <prosody>, and <emphasis> for natural, understandable speech.
 */
export function buildSsml(text: string, options: SsmlOptions = {}): string {
  const rate = options.rate ?? DEFAULT_RATE;
  const ratePercent = Math.round(rate * 100);
  const escaped = escapeSsml(text.trim());

  if (!escaped) return '<speak></speak>';

  let inner = escaped;

  // Apply emphasis on specified phrases (case-insensitive replace)
  if (options.emphasis?.length) {
    for (const phrase of options.emphasis) {
      const safe = escapeSsml(phrase);
      if (!safe) continue;
      const re = new RegExp(`(${safe.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
      inner = inner.replace(re, '<emphasis level="strong">$1</emphasis>');
    }
  }

  // Optional say-as for numbers (e.g. "1" spoken as "one")
  if (options.sayAs === 'digits') {
    inner = inner.replace(/\d+/g, (n) => `<say-as interpret-as="digits">${n}</say-as>`);
  }

  const prosodyAttrs: string[] = [`rate="${ratePercent}%"`];
  if (options.pitch) prosodyAttrs.push(`pitch="${options.pitch}"`);

  let wrap = `<prosody ${prosodyAttrs.join(' ')}>${inner}</prosody>`;
  if (options.breakAfter) {
    wrap += `<break time="${options.breakAfter}"/>`;
  }

  return `<speak>${wrap}</speak>`;
}

/**
 * Build SSML for a short announcement (e.g. button label, navigation).
 * Uses consistent rate and optional brief pause.
 */
export function buildAnnouncementSsml(text: string, rate?: number): string {
  return buildSsml(text, { rate: rate ?? DEFAULT_RATE, breakAfter: '200ms' });
}
