export function ApiStatus({
  online,
  failed,
}: {
  online: boolean;
  failed: boolean;
}) {
  if (!online && !failed) {
    return (
      <div className="flex items-center gap-2.5 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-500 shadow-sm">
        <span className="relative flex h-2.5 w-2.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-75" />
          <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-amber-400" />
        </span>
        Verbinding controleren...
      </div>
    );
  }

  if (online) {
    return (
      <div className="flex items-center gap-2.5 rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-medium text-emerald-700 shadow-sm">
        <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
        FastAPI verbonden
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2.5 rounded-full border border-red-200 bg-red-50 px-4 py-2 text-sm font-medium text-red-600 shadow-sm">
      <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
      FastAPI niet beschikbaar
    </div>
  );
}
