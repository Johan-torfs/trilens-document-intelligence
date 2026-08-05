"use client";

import type { AnalysisResponse, SearchResult } from "@/lib/api";
import { getApiUrl } from "@/lib/api";
import { ANALYSIS_QUESTIONS } from "@/lib/constants";
import { AnalysisResultPanel } from "@/components/AnalysisResultPanel";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { Score } from "@/components/ui/Score";
import { Spinner } from "@/components/ui/Spinner";

type SearchResultCardProps = {
  result: SearchResult;
  question: string;
  analysis?: AnalysisResponse;
  error?: string;
  isAnalyzing: boolean;
  onQuestionChange: (question: string) => void;
  onAnalyze: () => Promise<void>;
};

export function SearchResultCard({
  result,
  question,
  analysis,
  error,
  isAnalyzing,
  onQuestionChange,
  onAnalyze,
}: SearchResultCardProps) {
  return (
    <article className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm transition hover:shadow-md">
      <div className="grid lg:grid-cols-[300px_1fr]">
        {/* Document preview */}
        <div className="relative flex min-h-72 items-center justify-center bg-slate-100 p-5">
          <span className="absolute z-10 left-3 top-3 rounded-full bg-slate-900/75 px-2.5 py-1 font-mono text-xs font-medium text-white backdrop-blur-sm">
            #{result.rank}
          </span>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={getApiUrl(result.image_url)}
            alt={`Document ${result.document_id}`}
            className="max-h-105 max-w-full object-contain drop-shadow-sm"
          />
        </div>

        {/* Document info and analysis */}
        <div className="flex flex-col p-6">
          {/* Title row */}
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.15em] text-indigo-600">
                {result.document_type}
              </p>
              <p className="mt-1 break-all text-xs text-slate-400">
                {result.document_id}
              </p>
            </div>
            <span className="shrink-0 rounded-full bg-slate-900 px-3 py-1 font-mono text-sm font-semibold text-white">
              {Math.round(result.final_score * 100)}%
            </span>
          </div>

          {/* Score grid */}
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Score label="Final" value={result.final_score} />
            <Score label="Visual" value={result.visual_score} />
            <Score label="Text" value={result.text_score} />
            <Score label="Lexical" value={result.fts_score} />
          </div>

          {/* Analysis section */}
          <div className="mt-6 flex-1 rounded-xl border border-slate-200 bg-slate-50 p-4">
            <div className="mb-1 flex items-center gap-2">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                className="h-4 w-4 text-slate-500"
                aria-hidden="true"
              >
                <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                <path
                  fillRule="evenodd"
                  d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                  clipRule="evenodd"
                />
              </svg>
              <h4 className="text-sm font-semibold text-slate-800">
                Experimentele analyse
              </h4>
            </div>
            <p className="mb-4 text-xs text-slate-500">
              OpenFlamingo kan details verkeerd interpreteren of verzinnen.
            </p>

            <div className="flex flex-col gap-3 sm:flex-row">
              <select
                value={question}
                onChange={(event) => {
                  onQuestionChange(event.target.value);
                }}
                className="min-w-0 flex-1 rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
              >
                {ANALYSIS_QUESTIONS.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>

              <button
                type="button"
                disabled={isAnalyzing}
                onClick={() => void onAnalyze()}
                className="flex min-h-10 shrink-0 items-center justify-center rounded-xl bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 sm:min-w-40"
              >
                {isAnalyzing ? (
                  <Spinner label="Analyseren..." compact />
                ) : (
                  "Analyse uitvoeren"
                )}
              </button>
            </div>

            {error && <ErrorMessage message={error} />}
            {analysis && <AnalysisResultPanel analysis={analysis} />}
          </div>
        </div>
      </div>
    </article>
  );
}
