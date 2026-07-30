import type { AnalysisResponse, SearchResponse } from "@/lib/api";
import { ANALYSIS_QUESTIONS } from "@/lib/constants";
import { SearchResultCard } from "@/components/SearchResultCard";
import { SmallMetric } from "@/components/ui/SmallMetric";

type SearchResultsProps = {
  response: SearchResponse;
  questions: Record<string, string>;
  analyses: Record<string, AnalysisResponse>;
  errors: Record<string, string>;
  analyzingDocumentId: string | null;
  onQuestionChange: (documentId: string, question: string) => void;
  onAnalyze: (documentId: string) => Promise<void>;
};

export function SearchResults({
  response,
  questions,
  analyses,
  errors,
  analyzingDocumentId,
  onQuestionChange,
  onAnalyze,
}: SearchResultsProps) {
  return (
    <section className="mt-10">
      {/* Results header */}
      <div className="mb-6 flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-indigo-600">
            Search response
          </p>
          <h2 className="mt-1 text-2xl font-semibold text-slate-900">
            {response.results.length}{" "}
            {response.results.length === 1 ? "resultaat" : "resultaten"}
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Zoekopdracht:{" "}
            <span className="font-medium text-slate-700">
              &ldquo;{response.query}&rdquo;
            </span>
          </p>
        </div>

        <div className="flex gap-3">
          <SmallMetric
            label="Ranking"
            value={response.ranking_mode.toUpperCase()}
          />
          <SmallMetric
            label="Runtime"
            value={`${response.duration_ms.toFixed(0)} ms`}
          />
        </div>
      </div>

      {/* Empty state */}
      {response.results.length === 0 ? (
        <div className="flex flex-col items-center rounded-2xl border border-slate-200 bg-white py-16 text-center shadow-sm">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="h-10 w-10 text-slate-300"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9zm3.75 11.625a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z"
            />
          </svg>
          <p className="mt-4 text-sm font-medium text-slate-600">
            Geen overeenkomende documenten gevonden.
          </p>
          <p className="mt-1 text-xs text-slate-400">
            Probeer een andere zoekopdracht of filter.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {response.results.map((result) => (
            <SearchResultCard
              key={result.document_id}
              result={result}
              question={questions[result.document_id] ?? ANALYSIS_QUESTIONS[0]}
              analysis={analyses[result.document_id]}
              error={errors[result.document_id]}
              isAnalyzing={analyzingDocumentId === result.document_id}
              onQuestionChange={(question) => {
                onQuestionChange(result.document_id, question);
              }}
              onAnalyze={() => onAnalyze(result.document_id)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
