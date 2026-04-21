import { DuplicateGroup, Sample } from "@/lib/types";

export default function DuplicatesGallery({
  duplicates, samples, limit = 6,
}: {
  duplicates: DuplicateGroup[];
  samples: Sample[];
  limit?: number;
}) {
  const byToken = new Map(samples.map((s) => [s.sample_token, s]));
  const top = duplicates.slice(0, limit);
  if (top.length === 0)
    return <p className="text-slate-400 text-sm">No near-duplicate groups found.</p>;
  return (
    <div className="space-y-4">
      {top.map((g) => (
        <div key={g.representative} className="border border-line rounded p-3 bg-panel/50">
          <div className="flex items-center justify-between mb-2 text-xs">
            <span className="text-accent">
              group of {g.members.length} · mean cos = {g.mean_similarity.toFixed(3)}
            </span>
            <span className="text-slate-500">rep: {g.representative}</span>
          </div>
          <div className="flex gap-2 overflow-x-auto">
            {g.members.slice(0, 10).map((tok) => {
              const s = byToken.get(tok);
              return (
                <div key={tok} className="flex-shrink-0">
                  <img
                    src={s?.thumb ?? ""}
                    alt={tok}
                    className={`w-24 h-24 object-cover rounded border ${
                      tok === g.representative ? "border-accent" : "border-line"
                    }`}
                  />
                  <div className="text-[10px] text-slate-500 mt-1 truncate w-24">
                    {s?.scene_name ?? tok}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
