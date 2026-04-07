import path from "node:path";

import { loadAuditEntries, saveAuditEntry } from "./audit_store";
import { DEFAULT_EXPERIMENTS_ROOT, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_REVIEW_ROOT, REPO_ROOT, DATA_ROOT, REVIEW_STATIC_ROOT } from "./config";
import { listExperiments, loadExperiment, resolveImagePath } from "./loader";
import type { AuditEntry } from "./types";

type ServerOptions = {
  host: string;
  port: number;
  experimentsRoot: string;
  reviewRoot: string;
  defaultExperiment: string | null;
};

const options = parseArgs(Bun.argv.slice(2));

const server = Bun.serve({
  hostname: options.host,
  port: options.port,
  fetch(req) {
    return handleRequest(req, options);
  },
});

console.log(`SGOCR review app listening on http://${server.hostname}:${server.port}`);
if (options.defaultExperiment) {
  console.log(`Default experiment: ${options.defaultExperiment}`);
}

async function handleRequest(req: Request, options: ServerOptions): Promise<Response> {
  const url = new URL(req.url);
  if (req.method === "GET" && url.pathname === "/") {
    return serveStatic("index.html", "text/html; charset=utf-8");
  }
  if (req.method === "GET" && url.pathname === "/app.js") {
    return serveStatic("app.js", "application/javascript; charset=utf-8");
  }
  if (req.method === "GET" && url.pathname === "/styles.css") {
    return serveStatic("styles.css", "text/css; charset=utf-8");
  }
  if (req.method === "GET" && url.pathname === "/api/session") {
    const experiments = listExperiments({ repoRoot: REPO_ROOT, dataRoot: DATA_ROOT, experimentsRoot: options.experimentsRoot });
    const experimentName = url.searchParams.get("experiment") || options.defaultExperiment || experiments[0]?.name || null;
    if (!experimentName) {
      return json({ experiments, experiment: null, samples: [], audit: {} });
    }
    const { experiment, samples } = loadExperiment(experimentName, { repoRoot: REPO_ROOT, dataRoot: DATA_ROOT, experimentsRoot: options.experimentsRoot });
    const audit = loadAuditEntries(options.reviewRoot, experimentName);
    return json({ experiments, experiment, samples, audit, reviewRoot: path.join(options.reviewRoot, experimentName) });
  }
  if (req.method === "POST" && url.pathname === "/api/audit") {
    const experimentName = url.searchParams.get("experiment");
    if (!experimentName) {
      return json({ error: "Missing experiment" }, 400);
    }
    const body = (await req.json()) as AuditEntry;
    const audit = saveAuditEntry(options.reviewRoot, experimentName, body);
    return json({ ok: true, audit });
  }
  if ((req.method === "GET" || req.method === "HEAD") && url.pathname === "/api/image") {
    const imagePath = url.searchParams.get("path");
    if (!imagePath) {
      return json({ error: "Missing image path" }, 400);
    }
    try {
      const resolved = resolveImagePath(imagePath, { repoRoot: REPO_ROOT, dataRoot: DATA_ROOT });
      return new Response(Bun.file(resolved));
    } catch (error) {
      return json({ error: error instanceof Error ? error.message : "Invalid image path" }, 400);
    }
  }
  if (req.method === "GET" && url.pathname === "/api/health") {
    return json({ ok: true });
  }
  return new Response("Not found", { status: 404 });
}

function parseArgs(argv: string[]): ServerOptions {
  const result: ServerOptions = {
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    experimentsRoot: DEFAULT_EXPERIMENTS_ROOT,
    reviewRoot: DEFAULT_REVIEW_ROOT,
    defaultExperiment: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const value = argv[index + 1];
    if (arg === "--port" && value) {
      result.port = Number(value);
      index += 1;
    } else if (arg === "--host" && value) {
      result.host = value;
      index += 1;
    } else if (arg === "--experiments-root" && value) {
      result.experimentsRoot = path.resolve(value);
      index += 1;
    } else if (arg === "--review-root" && value) {
      result.reviewRoot = path.resolve(value);
      index += 1;
    } else if (arg === "--experiment" && value) {
      result.defaultExperiment = value;
      index += 1;
    }
  }
  return result;
}

function serveStatic(fileName: string, contentType: string): Response {
  return new Response(Bun.file(path.join(REVIEW_STATIC_ROOT, fileName)), {
    headers: {
      "content-type": contentType,
      "cache-control": "no-store",
    },
  });
}

function json(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}
