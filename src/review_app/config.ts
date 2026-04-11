import path from "node:path";
import { realpathSync } from "node:fs";

const HERE = import.meta.dir;
export const REPO_ROOT = realpathSync(path.resolve(HERE, "..", "..", ".."));
export const DATA_ROOT = realpathSync(path.join(REPO_ROOT, "data"));
export const REVIEW_STATIC_ROOT = path.join(HERE, "public");

export const DEFAULT_EXPERIMENTS_ROOT = path.join(DATA_ROOT, "ocr_spatial_qa", "final");
export const DEFAULT_REVIEW_ROOT = path.join(DATA_ROOT, "ocr_spatial_qa", "review");
export const DEFAULT_PORT = Number(process.env.SGOCR_REVIEW_PORT || "3187");
export const DEFAULT_HOST = process.env.SGOCR_REVIEW_HOST || "127.0.0.1";
