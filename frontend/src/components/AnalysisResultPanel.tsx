import type { AnalysisResponse } from "@/lib/api";

export function AnalysisResultPanel({
  analysis,
}: {
  analysis: AnalysisResponse;
}) {
  return (
    <div className="mt-5 overflow-hidden rounded-xl border border-slate-200 bg-white">
      {/* Result header */}
      <div className="flex items-center justify-between gap-3 border-b border-slate-100 px-4 py-3">
        <p className="text-sm font-semibold text-slate-800">
          Modelgegenereerde analyse
        </p>
        <span className="rounded-full bg-slate-100 px-3 py-1 font-mono text-xs text-slate-500">
          {analysis.model_duration_ms.toFixed(0)} ms
        </span>
      </div>

      <div className="p-4">
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
          {analysis.text}
        </p>

        {/* Source notice */}
        <div className="mt-4 flex gap-2.5 rounded-lg border border-blue-200 bg-blue-50 p-3">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className="mt-0.5 h-4 w-4 shrink-0 text-blue-500"
            aria-hidden="true"
          >
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z"
              clipRule="evenodd"
            />
          </svg>
          <p className="text-xs text-blue-800">
            Experimentele OpenFlamingo-output. Verifieer het resultaat altijd.
          </p>
        </div>

        <p className="mt-3 text-xs text-slate-400">{analysis.model_name}</p>
      </div>
    </div>
  );
}
