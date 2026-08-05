"use client";

import { type FormEvent } from "react";
import type { IndexDocumentResponse } from "@/lib/api";
import { DOCUMENT_TYPES } from "@/lib/constants";
import { Detail } from "@/components/ui/Detail";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { SmallMetric } from "@/components/ui/SmallMetric";
import { Spinner } from "@/components/ui/Spinner";

type UploadPanelProps = {
  file: File | null;
  documentType: string;
  result: IndexDocumentResponse | null;
  error: string | null;
  isLoading: boolean;
  onFileChange: (file: File | null) => void;
  onDocumentTypeChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
};

export function UploadPanel({
  file,
  documentType,
  result,
  error,
  isLoading,
  onFileChange,
  onDocumentTypeChange,
  onSubmit,
}: UploadPanelProps) {
  return (
    <section className="flex flex-col rounded-2xl border border-slate-200 bg-white shadow-sm">
      {/* Panel header */}
      <div className="border-b border-slate-100 px-6 py-5">
        <p className="text-xs font-semibold uppercase tracking-[0.15em] text-indigo-600">
          Indexing
        </p>
        <h2 className="mt-1 text-xl font-semibold text-slate-900">
          Upload document
        </h2>
      </div>

      <div className="p-6">
        <form onSubmit={onSubmit} className="space-y-5">
          {/* File picker */}
          <div>
            <p className="mb-2 block text-sm font-medium text-slate-700">
              Afbeelding
            </p>
            <label className="group relative block cursor-pointer">
              <div
                className={`flex min-h-22 flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-5 text-center transition-colors ${
                  file
                    ? "border-indigo-400 bg-indigo-50"
                    : "border-slate-300 bg-slate-50 group-hover:border-indigo-400 group-hover:bg-indigo-50"
                }`}
              >
                {file ? (
                  <>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      className="h-6 w-6 text-indigo-500"
                      aria-hidden="true"
                    >
                      <path d="M3 3.5A1.5 1.5 0 014.5 2h6.879a1.5 1.5 0 011.06.44l4.122 4.12A1.5 1.5 0 0117 7.622V16.5a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 013 16.5v-13z" />
                    </svg>
                    <p className="mt-1.5 max-w-full truncate px-4 text-sm font-medium text-indigo-700">
                      {file.name}
                    </p>
                    <p className="text-xs text-slate-500">
                      Klik om te wijzigen
                    </p>
                  </>
                ) : (
                  <>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                      className="h-6 w-6 text-slate-400 transition-colors group-hover:text-indigo-500"
                      aria-hidden="true"
                    >
                      <path d="M9.25 13.25a.75.75 0 001.5 0V4.636l2.955 3.129a.75.75 0 001.09-1.03l-4.25-4.5a.75.75 0 00-1.09 0l-4.25 4.5a.75.75 0 101.09 1.03L9.25 4.636v8.614z" />
                      <path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" />
                    </svg>
                    <p className="mt-1.5 text-sm text-slate-600">
                      <span className="font-semibold text-indigo-600">
                        Kies een afbeelding
                      </span>{" "}
                      of sleep hier naartoe
                    </p>
                    <p className="text-xs text-slate-400">PNG, JPG tot 10 MB</p>
                  </>
                )}
              </div>
              <input
                type="file"
                accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                onChange={(event) => {
                  onFileChange(event.target.files?.[0] ?? null);
                }}
                className="absolute inset-0 cursor-pointer opacity-0"
              />
            </label>
          </div>

          {/* Document type */}
          <div>
            <label
              htmlFor="upload-doc-type"
              className="mb-2 block text-sm font-medium text-slate-700"
            >
              Documenttype
            </label>
            <select
              id="upload-doc-type"
              value={documentType}
              onChange={(event) => {
                onDocumentTypeChange(event.target.value);
              }}
              className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
            >
              {DOCUMENT_TYPES.map(({ value, label }) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="flex min-h-11 w-full items-center justify-center rounded-xl bg-indigo-600 px-5 py-2.5 font-medium text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? (
              <Spinner label="Verwerken..." compact />
            ) : (
              "Document verwerken"
            )}
          </button>
        </form>

        {error && <ErrorMessage message={error} />}
      </div>

      {/* Upload result */}
      {result && (
        <div className="border-t border-slate-100 px-6 pb-6">
          <div className="mb-4 flex items-center gap-2 pt-5">
            <div className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-100">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                className="h-3 w-3 text-emerald-600"
                aria-hidden="true"
              >
                <path
                  fillRule="evenodd"
                  d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <p className="text-sm font-medium text-slate-700">
              Document geïndexeerd
            </p>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <SmallMetric
              label="Zoekbaar"
              value={result.is_searchable ? "Ja" : "Nee"}
            />
            <SmallMetric label="OCR" value={result.has_ocr ? "Ja" : "Nee"} />
            <SmallMetric
              label="Runtime"
              value={`${result.duration_ms.toFixed(0)} ms`}
            />
          </div>

          <dl className="mt-4 space-y-2.5 rounded-xl bg-slate-50 p-4 ring-1 ring-inset ring-slate-200">
            <Detail label="Document-ID" value={result.document_id} />
            <Detail label="Type" value={result.document_type} />
            {result.classification_confidence !== null && (
              <Detail
                label="Zekerheid detectie"
                value={`${(result.classification_confidence * 100).toFixed(0)}%`}
              />
            )}
          </dl>

          {result.reused_document && (
            <p className="mt-3 rounded-xl border border-blue-200 bg-blue-50 p-3 text-sm text-blue-700">
              Bestaande artifacts zijn hergebruikt.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
