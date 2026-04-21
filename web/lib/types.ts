export type Sample = {
  sample_token: string;
  scene_name: string;
  weather: string;
  time_of_day: string;
  location: string;
  ego_speed_mps: number;
  thumb: string;
  x: number;
  y: number;
};

export type DuplicateGroup = {
  representative: string;
  members: string[];
  mean_similarity: number;
};

export type Outlier = {
  sample_token: string;
  knn_distance: number;
  rank: number;
};

export type Cluster = {
  cluster_id: number;
  label: string;
  size: number;
  members: string[];
  centroid_2d: [number, number];
  sample_2d: [string, number, number][];
};

export type LineageRecord = {
  run_id: string;
  timestamp_utc: string;
  git_sha: string | null;
  python_version: string;
  encoder_backend: string;
  source: string;
  n_input_samples: number;
  n_curated_samples: number;
  n_duplicate_groups: number;
  n_outliers: number;
  n_clusters: number;
  duplicate_threshold: number;
  target_size: number;
  manifest_sha256: string;
  extras: Record<string, unknown>;
};

export type Summary = {
  source: string;
  encoder_backend: string;
  vector_backend: string;
  n_input_samples: number;
  n_scenes: number;
  n_duplicate_groups: number;
  n_samples_in_duplicate_groups: number;
  duplicate_rate_pct: number;
  n_outliers: number;
  n_clusters: number;
  n_curated: number;
  duplicate_threshold: number;
  target_size: number;
  headline_metric: string;
};
