export function Spinner({
  label,
  compact = false,
}: {
  label: string;
  compact?: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-2.5">
      <span
        className={`animate-spin rounded-full border-2 border-current border-t-transparent ${
          compact ? "h-4 w-4" : "h-5 w-5"
        }`}
      />
      <span>{label}</span>
    </span>
  );
}
