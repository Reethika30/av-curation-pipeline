import {
  getClusters,
  getDuplicates,
  getLineage,
  getOutliers,
  getSamples,
  getSummary,
} from "@/lib/data";
import StatCard from "@/components/StatCard";
import ClusterScatter from "@/components/ClusterScatter";
import DuplicatesGallery from "@/components/DuplicatesGallery";
import OutliersGallery from "@/components/OutliersGallery";
import LineageTimeline from "@/components/LineageTimeline";

export const dynamic = "force-static";

export default async function HomePage() {
  const [summary, samples, dups, outs, clusters, lineage] = await Promise.all([
    getSummary(),
    getSamples(),
    getDuplicates(),
    getOutliers(),
    getClusters(),
    getLineage(),
  ]);

  if (!summary) {
    return (
      <main className="max-w-3xl mx-auto p-10">
        <h1 className="text-2xl text-accent mb-4">av-curation-pipeline</h1>
        <p className="text-slate-300">
          No precomputed data found. From the repo root run:
        </p>
        <pre className="bg-panel border border-line rounded p-3 mt-3 text-sm overflow-x-auto">
{`pip install -e .[dev]
python -m precompute.run --source synthetic --n 400`}
        </pre>
      </main>
    );
  }

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 py-10 space-y-12">
      <header>
        <div className="text-xs text-slate-500 uppercase tracking-widest">
          multimodal curation pipeline · autonomous driving
        </div>
        <h1 className="text-3xl sm:text-4xl mt-2 text-slate-100">
          From <span className="text-accent">{summary.n_input_samples.toLocaleString()}</span> raw frames
          to <span className="text-good">{summary.n_curated.toLocaleString()}</span> high-value training samples
        </h1>
        <p className="text-slate-400 mt-3 max-w-3xl">
          {summary.headline_metric}
        </p>
        <div className="text-[11px] text-slate-600 mt-2">
          source: {summary.source} · encoder: {summary.encoder_backend} ·
          vector store: {summary.vector_backend} ·
          dup-threshold cos ≥ {summary.duplicate_threshold}
        </div>
      </header>

      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="input frames" value={summary.n_input_samples.toLocaleString()}
                  hint={`${summary.n_scenes} scenes`} />
        <StatCard label="near-duplicate rate"
                  value={`${summary.duplicate_rate_pct}%`}
                  hint={`${summary.n_duplicate_groups} groups`} />
        <StatCard label="failure-mode clusters" value={summary.n_clusters}
                  hint="HDBSCAN on UMAP(CLIP)" />
        <StatCard label="curated set" value={summary.n_curated}
                  hint={`target ${summary.target_size}`} />
      </section>

      <section>
        <h2 className="text-xl text-slate-200 mb-3">
          Embedding map · failure-mode clusters
        </h2>
        <p className="text-slate-400 text-sm mb-4 max-w-3xl">
          UMAP projection of CLIP image embeddings. Clusters are HDBSCAN-discovered
          and labeled by nearest CLIP text prompt — the model literally tells us
          what each blob represents.
        </p>
        <ClusterScatter samples={samples} clusters={clusters} />
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <h2 className="text-xl text-slate-200 mb-3">Near-duplicate groups</h2>
          <p className="text-slate-400 text-sm mb-4">
            DINOv2 cosine ≥ {summary.duplicate_threshold}. Representative is highlighted.
          </p>
          <DuplicatesGallery duplicates={dups} samples={samples} />
        </div>
        <div>
          <h2 className="text-xl text-slate-200 mb-3">Edge cases / outliers</h2>
          <p className="text-slate-400 text-sm mb-4">
            Top-percentile k-NN distance in CLIP space — frames the model finds
            unusual.
          </p>
          <OutliersGallery outliers={outs} samples={samples} />
        </div>
      </section>

      <section>
        <h2 className="text-xl text-slate-200 mb-3">Lineage</h2>
        <p className="text-slate-400 text-sm mb-4 max-w-3xl">
          Every curation run is recorded with git SHA, encoder backend,
          parameters and a manifest hash so any downstream training run is
          fully reproducible.
        </p>
        <LineageTimeline records={lineage} />
      </section>

      <footer className="border-t border-line pt-6 text-xs text-slate-600">
        Built with PyArrow · DuckDB · DINOv2 · CLIP · UMAP · HDBSCAN
        · FastAPI · Next.js · DVC. Source code on GitHub.
      </footer>
    </main>
  );
}
