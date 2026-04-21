import { Outlier, Sample } from "@/lib/types";

export default function OutliersGallery({
  outliers, samples, limit = 12,
}: {
  outliers: Outlier[];
  samples: Sample[];
  limit?: number;
}) {
  const byToken = new Map(samples.map((s) => [s.sample_token, s]));
  const top = outliers.slice(0, limit);
  if (top.length === 0)
    return <p className="text-slate-400 text-sm">No outliers surfaced.</p>;
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
      {top.map((o) => {
        const s = byToken.get(o.sample_token);
        return (
          <div key={o.sample_token}
               className="border border-line rounded overflow-hidden bg-panel/50">
            <img src={s?.thumb ?? ""} alt={o.sample_token}
                 className="w-full aspect-square object-cover" />
            <div className="p-2 text-[11px]">
              <div className="text-warn">k-NN dist: {o.knn_distance.toFixed(3)}</div>
              <div className="text-slate-300 truncate">{s?.scene_name}</div>
              <div className="text-slate-500">{s?.weather} · {s?.time_of_day}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
