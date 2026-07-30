function scoreColors(value: number) {
  if (value >= 0.7) return { text: "text-emerald-600", bar: "bg-emerald-500" };
  if (value >= 0.4) return { text: "text-amber-600", bar: "bg-amber-500" };
  return { text: "text-slate-500", bar: "bg-slate-400" };
}

export function Score({ label, value }: { label: string; value: number }) {
  const colors = scoreColors(value);
  return (
    <div className="rounded-xl bg-slate-50 p-3 ring-1 ring-inset ring-slate-200">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className={`mt-1 font-mono text-base font-semibold ${colors.text}`}>
        {value.toFixed(3)}
      </p>
      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200">
        <div
          className={`h-full rounded-full transition-all duration-500 ${colors.bar}`}
          style={{ width: `${Math.min(value * 100, 100)}%` }}
        />
      </div>
    </div>
  );
}
