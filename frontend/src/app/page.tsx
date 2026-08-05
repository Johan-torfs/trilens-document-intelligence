"use client";

import { type FormEvent, useEffect, useState } from "react";

import {
  analyzeDocument,
  getHealth,
  searchDocuments,
  uploadDocument,
  type AnalysisResponse,
  type HealthResponse,
  type IndexDocumentResponse,
  type SearchResponse,
} from "@/lib/api";
import { ANALYSIS_QUESTIONS } from "@/lib/constants";
import { ApiStatus } from "@/components/ApiStatus";
import { SearchPanel } from "@/components/SearchPanel";
import { SearchResults } from "@/components/SearchResults";
import { UploadPanel } from "@/components/UploadPanel";
import { Spinner } from "@/components/ui/Spinner";

export default function HomePage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const [healthError, setHealthError] = useState(false);

  const [uploadFile, setUploadFile] = useState<File | null>(null);

  const [uploadDocumentType, setUploadDocumentType] = useState("");

  const [uploadResult, setUploadResult] =
    useState<IndexDocumentResponse | null>(null);

  const [uploadError, setUploadError] = useState<string | null>(null);

  const [isUploading, setIsUploading] = useState(false);

  const [query, setQuery] = useState("");
  const [searchDocumentType, setSearchDocumentType] = useState("");

  const [topK, setTopK] = useState(5);

  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);

  const [searchError, setSearchError] = useState<string | null>(null);

  const [isSearching, setIsSearching] = useState(false);

  const [selectedQuestions, setSelectedQuestions] = useState<
    Record<string, string>
  >({});

  const [analysisResults, setAnalysisResults] = useState<
    Record<string, AnalysisResponse>
  >({});

  const [analysisErrors, setAnalysisErrors] = useState<Record<string, string>>(
    {},
  );

  const [analyzingDocumentId, setAnalyzingDocumentId] = useState<string | null>(
    null,
  );

  useEffect(() => {
    async function loadHealth() {
      try {
        const response = await getHealth();

        setHealth(response);
        setHealthError(false);
      } catch {
        setHealth(null);
        setHealthError(true);
      }
    }

    void loadHealth();
  }, []);

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!uploadFile) {
      setUploadError("Selecteer eerst een afbeelding.");
      return;
    }

    setUploadError(null);
    setUploadResult(null);
    setIsUploading(true);

    try {
      const response = await uploadDocument(uploadFile, uploadDocumentType);

      setUploadResult(response);
    } catch (error) {
      setUploadError(getErrorMessage(error));
    } finally {
      setIsUploading(false);
    }
  }

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!query.trim()) {
      setSearchError("Voer eerst een zoekopdracht in.");
      return;
    }

    setSearchError(null);
    setSearchResult(null);
    setAnalysisResults({});
    setAnalysisErrors({});
    setIsSearching(true);

    try {
      const response = await searchDocuments({
        query: query.trim(),
        top_k: topK,
        document_type: searchDocumentType || null,
      });

      setSearchResult(response);
    } catch (error) {
      setSearchError(getErrorMessage(error));
    } finally {
      setIsSearching(false);
    }
  }

  async function handleAnalysis(documentId: string) {
    const question = selectedQuestions[documentId] ?? ANALYSIS_QUESTIONS[0];

    setAnalyzingDocumentId(documentId);

    setAnalysisErrors((current) => {
      const updated = { ...current };
      delete updated[documentId];
      return updated;
    });

    try {
      const response = await analyzeDocument(documentId, question);

      setAnalysisResults((current) => ({
        ...current,
        [documentId]: response,
      }));
    } catch (error) {
      setAnalysisErrors((current) => ({
        ...current,
        [documentId]: getErrorMessage(error),
      }));
    } finally {
      setAnalyzingDocumentId(null);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-7xl px-5 py-10 lg:px-8">
        {/* Header */}
        <header className="mb-10 flex flex-col gap-5 border-b border-slate-200 pb-8 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="h-4 w-4 text-white"
                  aria-hidden="true"
                >
                  <path d="M10.75 16.82A7.462 7.462 0 0115 15.5c.71 0 1.396.098 2.046.282A.75.75 0 0018 15.06v-11a.75.75 0 00-.546-.721A9.006 9.006 0 0015 3a8.963 8.963 0 00-4.25 1.065V16.82zM9.25 4.065A8.963 8.963 0 005 3c-.85 0-1.673.118-2.454.339A.75.75 0 002 4.06v11a.75.75 0 00.954.721A7.506 7.506 0 015 15.5c1.579 0 3.042.487 4.25 1.32V4.065z" />
                </svg>
              </div>
              <p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">
                Visual Document Intelligence
              </p>
            </div>
            <h1 className="mt-3 text-4xl font-bold tracking-tight text-slate-900">
              TriLens
            </h1>
            <p className="mt-2 max-w-xl text-sm text-slate-500">
              Indexeer documenten, zoek met CLIP of hybride ranking en voer
              experimentele visuele analyse uit.
            </p>
          </div>

          <ApiStatus online={health?.status === "ok"} failed={healthError} />
        </header>

        {/* Upload + Search panels */}
        <section className="grid gap-6 lg:grid-cols-2">
          <UploadPanel
            file={uploadFile}
            documentType={uploadDocumentType}
            result={uploadResult}
            error={uploadError}
            isLoading={isUploading}
            onFileChange={setUploadFile}
            onDocumentTypeChange={setUploadDocumentType}
            onSubmit={handleUpload}
          />
          <SearchPanel
            query={query}
            documentType={searchDocumentType}
            topK={topK}
            error={searchError}
            isLoading={isSearching}
            onQueryChange={setQuery}
            onDocumentTypeChange={setSearchDocumentType}
            onTopKChange={setTopK}
            onSubmit={handleSearch}
          />
        </section>

        {/* Searching indicator */}
        {isSearching && (
          <div className="mt-10 flex items-center justify-center rounded-2xl border border-slate-200 bg-white py-20 shadow-sm">
            <Spinner label="Documenten doorzoeken..." />
          </div>
        )}

        {/* Results */}
        {searchResult && !isSearching && (
          <SearchResults
            response={searchResult}
            questions={selectedQuestions}
            analyses={analysisResults}
            errors={analysisErrors}
            analyzingDocumentId={analyzingDocumentId}
            onQuestionChange={(documentId, question) => {
              setSelectedQuestions((current) => ({
                ...current,
                [documentId]: question,
              }));
            }}
            onAnalyze={handleAnalysis}
          />
        )}
      </div>
    </div>
  );
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Er is een onverwachte fout opgetreden.";
}
