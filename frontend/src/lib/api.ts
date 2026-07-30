const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type HealthResponse = {
  status: string;
  service: string;
};

export type IndexDocumentResponse = {
  document_id: string;
  document_type: string;
  original_filename: string;

  is_searchable: boolean;
  has_caption: boolean;
  fully_succeeded: boolean;
  reused_document: boolean;

  caption: string | null;
  embedding_model: string | null;
  caption_model: string | null;

  embedding_error: string | null;
  caption_error: string | null;

  duration_ms: number;
};

export type SearchRequest = {
  query: string;
  top_k: number;
  document_type?: string | null;
  use_hybrid_ranking: boolean;
};

export type SearchResult = {
  document_id: string;
  rank: number;

  final_score: number;
  clip_score: number;
  caption_score: number;
  metadata_score: number;

  caption: string | null;
  image_url: string;
  document_type: string;
};

export type SearchResponse = {
  query: string;
  top_k: number;
  document_type: string | null;
  ranking_mode: string;
  results: SearchResult[];
  duration_ms: number;
};

export type AnalysisResponse = {
  document_id: string;
  question: string;
  text: string;

  source: string;
  used_fallback: boolean;

  model_name: string;
  model_version: string | null;

  model_duration_ms: number;
  total_duration_ms: number;
};

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const body = (await response.json()) as {
        detail?: string;
      };

      if (body.detail) {
        message = body.detail;
      }
    } catch {
      // Response bevatte geen bruikbare JSON-fout.
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export function getApiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }

  return `${API_URL}${path}`;
}

export function getDocumentImageUrl(documentId: string): string {
  return getApiUrl(`/api/documents/${encodeURIComponent(documentId)}/image`);
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(getApiUrl("/api/health"), {
    cache: "no-store",
  });

  return parseResponse<HealthResponse>(response);
}

export async function uploadDocument(
  file: File,
  documentType: string,
): Promise<IndexDocumentResponse> {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("document_type", documentType);

  const response = await fetch(getApiUrl("/api/documents"), {
    method: "POST",
    body: formData,
  });

  return parseResponse<IndexDocumentResponse>(response);
}

export async function searchDocuments(
  request: SearchRequest,
): Promise<SearchResponse> {
  const response = await fetch(getApiUrl("/api/search"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  return parseResponse<SearchResponse>(response);
}

export async function analyzeDocument(
  documentId: string,
  question: string,
): Promise<AnalysisResponse> {
  const response = await fetch(
    getApiUrl(`/api/documents/${encodeURIComponent(documentId)}/analysis`),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    },
  );

  return parseResponse<AnalysisResponse>(response);
}
