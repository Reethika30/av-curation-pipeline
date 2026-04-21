import { promises as fs } from "fs";
import path from "path";
import {
  Cluster,
  DuplicateGroup,
  LineageRecord,
  Outlier,
  Sample,
  Summary,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

async function readJson<T>(file: string, fallback: T): Promise<T> {
  try {
    const buf = await fs.readFile(path.join(DATA_DIR, file), "utf8");
    return JSON.parse(buf) as T;
  } catch {
    return fallback;
  }
}

export const getSummary = () =>
  readJson<Summary | null>("summary.json", null);
export const getSamples = () => readJson<Sample[]>("samples.json", []);
export const getDuplicates = () =>
  readJson<DuplicateGroup[]>("duplicates.json", []);
export const getOutliers = () => readJson<Outlier[]>("outliers.json", []);
export const getClusters = () => readJson<Cluster[]>("clusters.json", []);
export const getLineage = () => readJson<LineageRecord[]>("lineage.json", []);
