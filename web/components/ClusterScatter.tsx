"use client";
import {
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Cluster, Sample } from "@/lib/types";
import { useMemo, useState } from "react";

const PALETTE = [
  "#7dd3fc", "#fbbf24", "#f87171", "#4ade80", "#c084fc",
  "#fb7185", "#34d399", "#facc15", "#60a5fa", "#a78bfa",
];

export default function ClusterScatter({
  samples, clusters,
}: { samples: Sample[]; clusters: Cluster[] }) {
  const [active, setActive] = useState<number | "all">("all");
  const tokenToCluster = useMemo(() => {
    const m = new Map<string, number>();
    clusters.forEach((c) => c.members.forEach((t) => m.set(t, c.cluster_id)));
    return m;
  }, [clusters]);

  const series = useMemo(() => {
    const groups = new Map<number, { name: string; data: any[] }>();
    samples.forEach((s) => {
      const cid = tokenToCluster.get(s.sample_token);
      if (cid === undefined) return;
      if (active !== "all" && cid !== active) return;
      const cluster = clusters.find((c) => c.cluster_id === cid);
      if (!groups.has(cid)) {
        groups.set(cid, { name: cluster?.label ?? `cluster ${cid}`, data: [] });
      }
      groups.get(cid)!.data.push({
        x: s.x, y: s.y,
        scene: s.scene_name, weather: s.weather, tod: s.time_of_day,
        thumb: s.thumb, token: s.sample_token,
      });
    });
    return Array.from(groups.entries()).sort((a, b) => a[0] - b[0]);
  }, [samples, clusters, tokenToCluster, active]);

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-2 text-xs">
        <button
          onClick={() => setActive("all")}
          className={`px-2 py-1 rounded border border-line ${
            active === "all" ? "bg-accent text-ink" : "text-slate-300 hover:border-accent"
          }`}
        >
          all clusters
        </button>
        {clusters.map((c, i) => (
          <button
            key={c.cluster_id}
            onClick={() => setActive(c.cluster_id)}
            className={`px-2 py-1 rounded border ${
              active === c.cluster_id
                ? "border-accent text-ink"
                : "border-line text-slate-300 hover:border-accent"
            }`}
            style={{
              background: active === c.cluster_id ? PALETTE[i % PALETTE.length] : "transparent",
            }}
            title={c.label}
          >
            <span className="inline-block w-2 h-2 rounded-full mr-1.5"
                  style={{ background: PALETTE[i % PALETTE.length] }} />
            {c.label} <span className="opacity-60">({c.size})</span>
          </button>
        ))}
      </div>
      <div className="h-[460px] w-full bg-ink rounded border border-line p-2">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 12, right: 12, bottom: 12, left: 12 }}>
            <XAxis type="number" dataKey="x" hide domain={["auto", "auto"]} />
            <YAxis type="number" dataKey="y" hide domain={["auto", "auto"]} />
            <ZAxis range={[36, 36]} />
            <Tooltip
              cursor={false}
              wrapperStyle={{ outline: "none" }}
              content={({ active: a, payload }) => {
                if (!a || !payload?.length) return null;
                const d: any = payload[0].payload;
                return (
                  <div className="bg-panel border border-line p-2 rounded text-xs">
                    <img src={d.thumb} alt="" className="w-32 h-32 object-cover mb-1 rounded" />
                    <div className="font-bold">{d.scene}</div>
                    <div className="text-slate-400">
                      {d.weather} · {d.tod}
                    </div>
                    <div className="text-slate-500">{d.token}</div>
                  </div>
                );
              }}
            />
            {series.map(([cid, s], i) => (
              <Scatter
                key={cid}
                name={s.name}
                data={s.data}
                fill={PALETTE[clusters.findIndex((c) => c.cluster_id === cid) % PALETTE.length]}
                fillOpacity={0.85}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
