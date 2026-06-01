import type { AoiBounds, AoiCatalog, ValidationReport, WorkbenchCatalog } from "./types";

async function requestJson<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });

  const text = await response.text();
  let payload: unknown;

  try {
    payload = text ? JSON.parse(text) : {};
  } catch (error) {
    throw new Error(
      `Expected JSON from ${path}, but received a non-JSON response. Is the config API running?`
    );
  }

  if (!response.ok) {
    const error = payload && typeof payload === "object" && "error" in payload
      ? String(payload.error)
      : `Request failed: ${response.status}`;
    throw new Error(error);
  }

  return payload as T;
}

export function fetchCatalog(): Promise<WorkbenchCatalog> {
  return requestJson<WorkbenchCatalog>("/api/catalog");
}

export function validateRunConfig(runConfig: unknown): Promise<ValidationReport> {
  return requestJson<ValidationReport>("/api/validate-run", {
    method: "POST",
    body: JSON.stringify({ run_config: runConfig })
  });
}

export function createAoiConfig(payload: {
  name: string;
  description?: string;
  crs: string;
  bounds: AoiBounds;
}): Promise<{ ok: boolean; aoi: AoiCatalog }> {
  return requestJson("/api/aoi-config", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function createProjectGrid(payload: {
  project_config: string;
  aoi_config: string;
  crs: string;
  resolution_m: number;
  overwrite?: boolean;
}): Promise<{ ok: boolean; grid_path: string }> {
  return requestJson("/api/grid", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function renderRunConfig(runConfig: unknown): Promise<{
  ok: boolean;
  validation: ValidationReport;
  yaml: string;
}> {
  return requestJson("/api/render-run", {
    method: "POST",
    body: JSON.stringify({ run_config: runConfig })
  });
}
