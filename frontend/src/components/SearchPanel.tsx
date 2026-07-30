"use client";

import { type FormEvent } from "react";
import { DOCUMENT_TYPES } from "@/lib/constants";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { Spinner } from "@/components/ui/Spinner";

type SearchPanelProps = {
  query: string;
  documentType: string;
  topK: number;
  useHybrid: boolean;
  error: string | null;
  isLoading: boolean;
  onQueryChange: (value: string) => void;
  onDocumentTypeChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onHybridChange: (value: boolean) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
};

export function SearchPanel({
  query,
  documentType,
  topK,
  useHybrid,
  error,
  isLoading,
  onQueryChange,
  onDocumentTypeChange,
  onTopKChange,
  onHybridChange,
  onSubmit,
}: SearchPanelProps) {
  return (
    <section className="flex flex-col rounded-2xl border border-slate-200 bg-white shadow-sm">
      {/* Panel header */}
      <div className="border-b border-slate-100 px-6 py-5">
        <p className="text-xs font-semibold uppercase tracking-[0.15em] text-indigo-600">
          Retrieval
        </p>
        <h2 className="mt-1 text-xl font-semibold text-slate-900">
          Search documents
        </h2>
      </div>

      <div className="p-6">
        <form onSubmit={onSubmit} className="space-y-5">
          {/* Search query with icon */}
          <div>
            <label
              htmlFor="search-query"
              className="mb-2 block text-sm font-medium text-slate-700"
            >
              Zoekopdracht
            </label>
            <div className="relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="h-4 w-4 text-slate-400"
                  aria-hidden="true"
                >
                  <path
                    fillRule="evenodd"
                    d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <input
                id="search-query"
                value={query}
                onChange={(event) => {
                  onQueryChange(event.target.value);
                }}
                placeholder="invoice with several product rows"
                className="w-full rounded-xl border border-slate-300 bg-white py-2.5 pl-10 pr-3 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
              />
            </div>
          </div>

          {/* Document type + top-k */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="search-doc-type"
                className="mb-2 block text-sm font-medium text-slate-700"
              >
                Documenttype
              </label>
              <select
                id="search-doc-type"
                value={documentType}
                onChange={(event) => {
                  onDocumentTypeChange(event.target.value);
                }}
                className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
              >
                {DOCUMENT_TYPES.map(({ value, label }) => (
                  <option key={label} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label
                htmlFor="search-topk"
                className="mb-2 block text-sm font-medium text-slate-700"
              >
                Top-k resultaten
              </label>
              <input
                id="search-topk"
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(event) => {
                  onTopKChange(Number(event.target.value));
                }}
                className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
              />
            </div>
          </div>

          {/* Hybrid ranking toggle */}
          <label className="flex cursor-pointer items-center gap-4 rounded-xl border border-slate-200 bg-slate-50 p-3.5 transition hover:bg-slate-100">
            <div className="relative shrink-0">
              <input
                type="checkbox"
                checked={useHybrid}
                onChange={(event) => {
                  onHybridChange(event.target.checked);
                }}
                className="sr-only"
              />
              <div
                className={`h-5 w-9 rounded-full transition-colors ${
                  useHybrid ? "bg-indigo-600" : "bg-slate-300"
                }`}
              />
              <div
                className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
                  useHybrid ? "translate-x-4" : "translate-x-0"
                }`}
              />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700">
                Hybride ranking
              </p>
              <p className="text-xs text-slate-500">
                Combineert CLIP- en captionscores
              </p>
            </div>
          </label>

          <button
            type="submit"
            disabled={isLoading}
            className="flex min-h-11 w-full items-center justify-center rounded-xl bg-indigo-600 px-5 py-2.5 font-medium text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? <Spinner label="Zoeken..." compact /> : "Zoeken"}
          </button>
        </form>

        {error && <ErrorMessage message={error} />}
      </div>
    </section>
  );
}
