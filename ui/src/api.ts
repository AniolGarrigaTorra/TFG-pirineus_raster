import type { ValidationReport, WorkbenchCatalog } from "./types";

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

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
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

