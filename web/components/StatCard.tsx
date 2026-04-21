export default function StatCard({
  label, value, hint,
}: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="border border-line rounded p-4 bg-panel/60">
      <div className="text-[11px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="text-2xl text-accent mt-1">{value}</div>
      {hint && <div className="text-[11px] text-slate-500 mt-1">{hint}</div>}
    </div>
  );
}
