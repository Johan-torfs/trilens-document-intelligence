export function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid gap-1 text-sm sm:grid-cols-[130px_1fr]">
      <dt className="font-medium text-slate-500">{label}</dt>
      <dd className="break-all text-slate-800">{value}</dd>
    </div>
  );
}
