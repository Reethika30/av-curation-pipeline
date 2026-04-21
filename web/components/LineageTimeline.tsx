import { LineageRecord } from "@/lib/types";

export default function LineageTimeline({ records }: { records: LineageRecord[] }) {
  if (records.length === 0)
    return <p className="text-slate-400 text-sm">No lineage runs recorded yet.</p>;
  const sorted = [...records].sort((a, b) =>
    a.timestamp_utc < b.timestamp_utc ? 1 : -1
  );
  return (
    <ol className="relative border-l border-line ml-3 space-y-4">
      {sorted.map((r) => (
        <li key={r.run_id} className="ml-4">
          <span className="absolute -left-1.5 mt-1.5 w-3 h-3 rounded-full bg-accent" />
          <div className="text-xs text-slate-500">{r.timestamp_utc}</div>
          <div className="font-bold text-accent">{r.run_id}</div>
          <div className="text-xs text-slate-400">
            git <span className="text-slate-200">{r.git_sha?.slice(0, 8) ?? "—"}</span>{" "}
            · py {r.python_version} · encoder{" "}
            <span className="text-slate-200">{r.encoder_backend}</span>
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {r.n_input_samples} → {r.n_curated_samples} samples ·{" "}
            {r.n_duplicate_groups} dup-groups · {r.n_outliers} outliers ·{" "}
            {r.n_clusters} clusters
          </div>
          <div className="text-[11px] text-slate-600 mt-1 break-all">
            sha256: {r.manifest_sha256.slice(0, 32)}…
          </div>
        </li>
      ))}
    </ol>
  );
}
