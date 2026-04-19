/**
 * Favorites API for the document microservice (GET/POST /favorites).
 */

const env = (import.meta as any).env;

const BASE =
  env?.VITE_API_URL_DOCUMENT || env?.VITE_API_URL || 'http://localhost:8000';
const PREFIX = env?.VITE_API_DOCUMENT_PREFIX ?? '';

function favoritesUrl() {
  return `${BASE}${PREFIX}/favorites`;
}

export interface FavoriteArticle {
  document_id: string;
  article_id: string;
  heading?: string;
  subheading?: string;
  body_preview?: string;
  /** Persisted summary from the document service / Mongo (used instead of re-summarizing). */
  summary?: string;
  /** Full article text stored in Mongo for Q&A without relying on in-memory /process state. */
  full_content?: string;
  resource_type?: string;
  created_at?: string;
  updated_at?: string;
}

export interface ListFavoritesResponse {
  count: number;
  favorites: FavoriteArticle[];
}

export interface AddFavoriteResponse {
  message?: string;
  favorite?: Record<string, unknown>;
}

async function parseErrorMessage(res: Response, fallback: string): Promise<string> {
  let msg = fallback;
  try {
    const data = await res.json();
    const detail = data?.detail;
    if (typeof detail === 'string') {
      msg = detail;
    } else if (detail != null) {
      msg = JSON.stringify(detail);
    }
  } catch {
    // ignore
  }
  return msg;
}

export async function listFavoriteArticles(): Promise<ListFavoritesResponse> {
  const res = await fetch(favoritesUrl(), { method: 'GET' });

  if (!res.ok) {
    const msg = await parseErrorMessage(
      res,
      `Could not load favorites (${res.status})`
    );
    throw new Error(msg);
  }

  return res.json();
}

export async function addFavoriteArticle(
  documentId: string,
  articleId: string
): Promise<AddFavoriteResponse> {
  if (!documentId?.trim() || !articleId?.trim()) {
    throw new Error('Document and article are required to save a favorite.');
  }

  const res = await fetch(favoritesUrl(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      document_id: documentId,
      article_id: articleId,
    }),
  });

  if (!res.ok) {
    const msg = await parseErrorMessage(
      res,
      `Could not save favorite (${res.status})`
    );
    throw new Error(msg);
  }

  return res.json();
}
