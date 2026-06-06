import { useEffect, useMemo, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { ReactNode } from "react";
import { createAoiConfig, createProjectGrid, fetchCatalog, renderRunConfig, saveRunConfig, validateRunConfig } from "./api";
import type {
  AoiCatalog,
  AoiBounds,
  CategoryFractionSelection,
  CustomAggregation,
  DatasetFeatureConfig,
  DatasetFeatureInput,
  DatasetFeatureOutput,
  DerivedFeatureConfig,
  DerivedInputQuery,
  FeatureSourceInput,
  SourceCatalog,
  SourceSelection,
  TemporalSelection,
  VariableCatalog,
  ValidationReport,
  WorkbenchCatalog
} from "./types";
import { renderYaml } from "./yaml";
import "./App.css";

const tabs = ["Project", "Sources", "Variables", "Temporal", "Derived", "Review"];
type StartMode = "menu" | "project" | "aoi" | "sources" | "tutorial";
type FeatureTemporalMode = "static" | "supplied_layers" | "aggregate" | "raw_slices" | "postprocess_aggregate";
type FeaturePickerStep = "origin" | "provider" | "source" | "variable" | "category" | "dimensions" | "temporal" | "resampling";
type InputOutputOption = {
  name: string;
  label: string;
  input: DatasetFeatureInput;
  suffix?: string;
  temporalKey?: string;
  dimensionKey?: string;
  variable?: VariableCatalog;
};
type InputBundle = {
  label: string;
  outputs: InputOutputOption[];
};
const backgroundManifestUrl = "/backgrounds/manifest.json";
const backgroundRotationMs = 5 * 60 * 1000;
const temporalDimensionKeys = new Set(["year", "years", "month", "months", "season", "seasons"]);
const pyreneesWgs84Envelope = { xmin: -2.8, xmax: 3.9, ymin: 41.0, ymax: 43.9 };
const aoiInitialMapZoom = 8;
const aoiMapViewport = { width: 1180, height: 690 };
type LonLatPoint = { lon: number; lat: number };
type ProjectedPoint = { x: number; y: number };

function normalizeBackgroundUrls(value: unknown) {
  if (!Array.isArray(value)) return [];

  const urls = value
    .filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    .map((item) => item.trim());

  return [...new Set(urls)];
}

function pickRandomBackground(urls: string[], currentUrl?: string) {
  if (urls.length === 0) return null;
  if (urls.length === 1) return urls[0];

  let nextUrl = currentUrl;
  for (let attempt = 0; attempt < 8 && nextUrl === currentUrl; attempt += 1) {
    nextUrl = urls[Math.floor(Math.random() * urls.length)];
  }
  return nextUrl ?? urls[0];
}

function preloadBackground(url: string) {
  return new Promise<string>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(url);
    image.onerror = () => reject(new Error(`Could not load background image: ${url}`));
    image.src = url;
  });
}

function BackgroundCredit() {
  return <div className="photo-credit">Photo: Felipe Valladares</div>;
}

function BackgroundImage({ src, variant }: { src: string; variant: "welcome" | "workbench" }) {
  return (
    <img
      className={`background-image background-image-${variant}`}
      src={src}
      alt=""
      aria-hidden="true"
      draggable={false}
    />
  );
}

function WorkbenchShell({ backgroundUrl, children }: { backgroundUrl: string; children: ReactNode }) {
  return (
    <div className="app-shell workbench-shell">
      <BackgroundImage src={backgroundUrl} variant="workbench" />
      {children}
    </div>
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function lonLatToPixel(lon: number, lat: number, zoom: number) {
  const sinLat = Math.sin((lat * Math.PI) / 180);
  const scale = 256 * 2 ** zoom;
  return {
    x: ((lon + 180) / 360) * scale,
    y: (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale
  };
}

function pixelToLonLat(x: number, y: number, zoom: number) {
  const scale = 256 * 2 ** zoom;
  const lon = (x / scale) * 360 - 180;
  const n = Math.PI - (2 * Math.PI * y) / scale;
  const lat = (180 / Math.PI) * Math.atan(Math.sinh(n));
  return { lon, lat };
}

function orderedBounds(bounds: AoiBounds): AoiBounds {
  return {
    xmin: Math.min(bounds.xmin, bounds.xmax),
    xmax: Math.max(bounds.xmin, bounds.xmax),
    ymin: Math.min(bounds.ymin, bounds.ymax),
    ymax: Math.max(bounds.ymin, bounds.ymax)
  };
}

function formatCoord(value: number) {
  return Number(value.toFixed(8));
}

function formatCrsCoord(value: number, crs: string) {
  return /4326/i.test(crs)
    ? Number(value.toFixed(8))
    : Number(value.toFixed(3));
}

function normalizeCrsCode(crs: string) {
  const value = crs.trim().toUpperCase();
  if (value === "4326" || value.endsWith(":4326")) return "EPSG:4326";
  if (value === "3035" || value.endsWith(":3035")) return "EPSG:3035";
  return value;
}

function canProjectToMap(crs: string) {
  return ["EPSG:4326", "EPSG:3035"].includes(normalizeCrsCode(crs));
}

function densifiedBoundsEdgePoints(bounds: AoiBounds, segments = 32): ProjectedPoint[] {
  const safeBounds = orderedBounds(bounds);
  const points: ProjectedPoint[] = [];
  const steps = Math.max(1, Math.round(segments));

  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    points.push({
      x: safeBounds.xmin + (safeBounds.xmax - safeBounds.xmin) * t,
      y: safeBounds.ymin
    });
  }
  for (let i = 1; i <= steps; i += 1) {
    const t = i / steps;
    points.push({
      x: safeBounds.xmax,
      y: safeBounds.ymin + (safeBounds.ymax - safeBounds.ymin) * t
    });
  }
  for (let i = 1; i <= steps; i += 1) {
    const t = i / steps;
    points.push({
      x: safeBounds.xmax - (safeBounds.xmax - safeBounds.xmin) * t,
      y: safeBounds.ymax
    });
  }
  for (let i = 1; i < steps; i += 1) {
    const t = i / steps;
    points.push({
      x: safeBounds.xmin,
      y: safeBounds.ymax - (safeBounds.ymax - safeBounds.ymin) * t
    });
  }

  return points;
}

function envelopeFromLonLatPoints(points: LonLatPoint[]): AoiBounds | null {
  if (points.length === 0) return null;
  return orderedBounds({
    xmin: Math.min(...points.map((point) => point.lon)),
    xmax: Math.max(...points.map((point) => point.lon)),
    ymin: Math.min(...points.map((point) => point.lat)),
    ymax: Math.max(...points.map((point) => point.lat))
  });
}

const laea3035 = (() => {
  const a = 6378137;
  const invF = 298.257222101;
  const f = 1 / invF;
  const e2 = 2 * f - f * f;
  const e = Math.sqrt(e2);
  const lat0 = 52 * Math.PI / 180;
  const lon0 = 10 * Math.PI / 180;
  const falseEasting = 4321000;
  const falseNorthing = 3210000;

  function q(phi: number) {
    const sinPhi = Math.sin(phi);
    return (1 - e2) * (
      sinPhi / (1 - e2 * sinPhi * sinPhi) -
      (1 / (2 * e)) * Math.log((1 - e * sinPhi) / (1 + e * sinPhi))
    );
  }

  const qp = q(Math.PI / 2);
  const beta0 = Math.asin(q(lat0) / qp);
  const rq = a * Math.sqrt(qp / 2);
  const sinBeta0 = Math.sin(beta0);
  const cosBeta0 = Math.cos(beta0);
  const m0 = Math.cos(lat0) / Math.sqrt(1 - e2 * Math.sin(lat0) ** 2);
  const d = (a * m0) / (rq * cosBeta0);

  function betaToGeodetic(beta: number) {
    const targetQ = qp * Math.sin(beta);
    let phi = beta;
    for (let i = 0; i < 8; i += 1) {
      const currentQ = q(phi);
      const delta = 1e-7;
      const derivative = (q(phi + delta) - q(phi - delta)) / (2 * delta);
      if (!Number.isFinite(derivative) || Math.abs(derivative) < 1e-12) break;
      phi -= (currentQ - targetQ) / derivative;
    }
    return phi;
  }

  return {
    forward(lon: number, lat: number) {
      const lambda = lon * Math.PI / 180;
      const phi = lat * Math.PI / 180;
      const beta = Math.asin(q(phi) / qp);
      const b = rq * Math.sqrt(2 / (1 + sinBeta0 * Math.sin(beta) + cosBeta0 * Math.cos(beta) * Math.cos(lambda - lon0)));
      return {
        x: falseEasting + b * d * Math.cos(beta) * Math.sin(lambda - lon0),
        y: falseNorthing + (b / d) * (cosBeta0 * Math.sin(beta) - sinBeta0 * Math.cos(beta) * Math.cos(lambda - lon0))
      };
    },
    inverse(x: number, y: number) {
      const xp = (x - falseEasting) / d;
      const yp = (y - falseNorthing) * d;
      const rho = Math.sqrt(xp * xp + yp * yp);
      if (rho < 1e-9) {
        return { lon: 10, lat: 52 };
      }
      const c = 2 * Math.asin(Math.min(1, rho / (2 * rq)));
      const beta = Math.asin(Math.cos(c) * sinBeta0 + (yp * Math.sin(c) * cosBeta0) / rho);
      const lambda = lon0 + Math.atan2(
        xp * Math.sin(c),
        rho * cosBeta0 * Math.cos(c) - yp * sinBeta0 * Math.sin(c)
      );
      return {
        lon: lambda * 180 / Math.PI,
        lat: betaToGeodetic(beta) * 180 / Math.PI
      };
    }
  };
})();

function pointToWgs84(x: number, y: number, crs: string): LonLatPoint | null {
  const normalized = normalizeCrsCode(crs);
  if (normalized === "EPSG:4326") return { lon: x, lat: y };
  if (normalized === "EPSG:3035") return laea3035.inverse(x, y);
  return null;
}

function pointFromWgs84(lon: number, lat: number, crs: string): ProjectedPoint | null {
  const normalized = normalizeCrsCode(crs);
  if (normalized === "EPSG:4326") return { x: lon, y: lat };
  if (normalized === "EPSG:3035") return laea3035.forward(lon, lat);
  return null;
}

function boundsFootprintToWgs84(bounds: AoiBounds, crs: string, segments = 48) {
  const points = densifiedBoundsEdgePoints(bounds, segments)
    .map((point) => pointToWgs84(point.x, point.y, crs))
    .filter((point): point is LonLatPoint => Boolean(point));
  return points.length > 0 ? points : null;
}

function boundsToWgs84(bounds: AoiBounds, crs: string) {
  const footprint = boundsFootprintToWgs84(bounds, crs);
  return footprint ? envelopeFromLonLatPoints(footprint) : null;
}

function boundsFromWgs84(bounds: AoiBounds, crs: string) {
  const corners = densifiedBoundsEdgePoints(bounds, 48)
    .map((point) => pointFromWgs84(point.x, point.y, crs))
    .filter((point): point is ProjectedPoint => Boolean(point));
  if (corners.length === 0) return null;
  return orderedBounds({
    xmin: Math.min(...corners.map((point) => point.x)),
    xmax: Math.max(...corners.map((point) => point.x)),
    ymin: Math.min(...corners.map((point) => point.y)),
    ymax: Math.max(...corners.map((point) => point.y))
  });
}

function gcd(left: number, right: number): number {
  let a = Math.abs(Math.round(left));
  let b = Math.abs(Math.round(right));
  while (b > 0) {
    const next = a % b;
    a = b;
    b = next;
  }
  return a || 1;
}

function lcm(values: number[]) {
  const positive = values.map((value) => Math.round(value)).filter((value) => value > 0);
  if (positive.length === 0) return 1;
  return positive.reduce((acc, value) => Math.abs(acc * value) / gcd(acc, value));
}

function resolutionStepForBounds(bounds: AoiBounds, crs: string, resolutionM: number) {
  if (/4326/i.test(crs)) {
    const midLat = ((bounds.ymin + bounds.ymax) / 2) * Math.PI / 180;
    const metresPerDegLat = 111_320;
    const metresPerDegLon = Math.max(1, 111_320 * Math.cos(midLat));
    return {
      x: resolutionM / metresPerDegLon,
      y: resolutionM / metresPerDegLat,
      approximate: true
    };
  }
  return { x: resolutionM, y: resolutionM, approximate: false };
}

function isMultipleOfStep(value: number, step: number) {
  if (!Number.isFinite(value) || !Number.isFinite(step) || step <= 0) return false;
  const ratio = value / step;
  const nearest = Math.round(ratio) * step;
  const tolerance = step < 1 ? step * 0.08 : 0.01;
  return Math.abs(value - nearest) <= tolerance;
}

function resolutionChecksForBounds(bounds: AoiBounds, crs: string, resolutions: number[]) {
  const safeBounds = orderedBounds(bounds);
  const width = safeBounds.xmax - safeBounds.xmin;
  const height = safeBounds.ymax - safeBounds.ymin;
  return resolutions.map((resolution) => {
    const step = resolutionStepForBounds(safeBounds, crs, resolution);
    return {
      resolution,
      widthOk: isMultipleOfStep(width, step.x),
      heightOk: isMultipleOfStep(height, step.y),
      approximate: step.approximate
    };
  });
}

function shiftIntoEnvelope(bounds: AoiBounds, envelope: AoiBounds) {
  let next = { ...bounds };
  const width = next.xmax - next.xmin;
  const height = next.ymax - next.ymin;
  if (width >= envelope.xmax - envelope.xmin) {
    next.xmin = envelope.xmin;
    next.xmax = envelope.xmax;
  } else {
    if (next.xmin < envelope.xmin) {
      next.xmax += envelope.xmin - next.xmin;
      next.xmin = envelope.xmin;
    }
    if (next.xmax > envelope.xmax) {
      next.xmin -= next.xmax - envelope.xmax;
      next.xmax = envelope.xmax;
    }
  }
  if (height >= envelope.ymax - envelope.ymin) {
    next.ymin = envelope.ymin;
    next.ymax = envelope.ymax;
  } else {
    if (next.ymin < envelope.ymin) {
      next.ymax += envelope.ymin - next.ymin;
      next.ymin = envelope.ymin;
    }
    if (next.ymax > envelope.ymax) {
      next.ymin -= next.ymax - envelope.ymax;
      next.ymax = envelope.ymax;
    }
  }
  return next;
}

function expandBoundsToResolutions(bounds: AoiBounds, crs: string, resolutions: number[]) {
  const safeBounds = orderedBounds(bounds);
  const commonResolution = lcm(resolutions);
  const step = resolutionStepForBounds(safeBounds, crs, commonResolution);
  const width = safeBounds.xmax - safeBounds.xmin;
  const height = safeBounds.ymax - safeBounds.ymin;
  const nextWidth = Math.max(step.x, Math.ceil(width / step.x) * step.x);
  const nextHeight = Math.max(step.y, Math.ceil(height / step.y) * step.y);
  const dx = (nextWidth - width) / 2;
  const dy = (nextHeight - height) / 2;
  let next = {
    xmin: safeBounds.xmin - dx,
    xmax: safeBounds.xmax + dx,
    ymin: safeBounds.ymin - dy,
    ymax: safeBounds.ymax + dy
  };
  if (/4326/i.test(crs)) {
    next = shiftIntoEnvelope(next, pyreneesWgs84Envelope);
  }
  return {
    xmin: formatCrsCoord(next.xmin, crs),
    xmax: formatCrsCoord(next.xmax, crs),
    ymin: formatCrsCoord(next.ymin, crs),
    ymax: formatCrsCoord(next.ymax, crs)
  };
}

function sourceVariables(source: SourceCatalog) {
  return [...(source.variables ?? []), ...(source.layers ?? [])];
}

function categoryClassValues(item: { value?: string | number; values?: Array<string | number> }) {
  if (Array.isArray(item.values) && item.values.length > 0) return item.values;
  return item.value !== undefined ? [item.value] : [];
}

function categoryClassToken(item: { name?: string; label?: string; value?: string | number; values?: Array<string | number> }) {
  const fallback = categoryClassValues(item).join("_");
  return sanitizeToken(item.name ?? item.label ?? fallback);
}

function sanitizeToken(value: string | number) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function humanizeId(value: string) {
  const acronyms: Record<string, string> = {
    cmip6: "CMIP6",
    clms: "CLMS",
    corine: "CORINE",
    dem: "DEM",
    glo30: "GLO-30",
    hrsi: "HRSI",
    igme: "IGME",
    brgm: "BRGM",
    pdca: "PDCA",
    epsg: "EPSG"
  };
  return value
    .replace(/[_-]+/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((word) => acronyms[word.toLowerCase()] ?? `${word.slice(0, 1).toUpperCase()}${word.slice(1)}`)
    .join(" ");
}

function sourceDisplayName(source: SourceCatalog) {
  return source.title ?? humanizeId(source.product ?? source.id);
}

function sourceShortName(source: SourceCatalog) {
  return source.id
    .replace(/^worldclim_/, "WC ")
    .replace(/^copernicus_/, "CLMS ")
    .replace(/^igme_brgm_/, "IGME-BRGM ")
    .replace(/^pdca_/, "PDCA ")
    .replace(/_/g, " ");
}

function sourceOfficialUrl(source: SourceCatalog) {
  if (source.official_url) return source.official_url;
  if (source.page_url) return source.page_url;
  if (source.documentation_url) return source.documentation_url;
  if (source.article_url) return source.article_url;
  if (source.doi) return `https://doi.org/${source.doi}`;
  return undefined;
}

function MetaChip({ label, value }: { label: string; value?: string | number }) {
  if (value === undefined || value === null || value === "") return null;
  return (
    <span>
      <strong>{label}</strong>
      {value}
    </span>
  );
}

function InfoTip({ text }: { text: string }) {
  return (
    <span className="info-tip" tabIndex={0} aria-label={text}>
      i
      <span role="tooltip">{text}</span>
    </span>
  );
}

function providerDisplayName(source: SourceCatalog) {
  return source.provider_title ?? humanizeId(source.provider);
}

function sourceAggregationNames(source: SourceCatalog) {
  return (source.aggregations ?? [])
    .map((item) => String(item.name ?? ""))
    .filter(Boolean);
}

function defaultRange(value?: [number, number], fallback: [number, number] = [1, 12]): [number, number] {
  return Array.isArray(value) && value.length === 2 ? [Number(value[0]), Number(value[1])] : fallback;
}

function sourceDimensionEntries(source: SourceCatalog) {
  return Object.entries(source.dimensions ?? {}).filter(
    ([key]) => !temporalDimensionKeys.has(key.toLowerCase())
  );
}

function sourceResolutionChoices(source: SourceCatalog | undefined) {
  if (!source) return [];
  const values = source.source_resolution_options && source.source_resolution_options.length > 0
    ? source.source_resolution_options
    : source.source_resolution
      ? [source.source_resolution]
      : [];
  return [...new Set(values.filter(Boolean))];
}

function parseYearBoundsFromToken(value: string | number) {
  const text = String(value);
  const years = [...text.matchAll(/\b(19|20)\d{2}\b/g)].map((match) => Number(match[0]));
  if (years.length === 0) return undefined;
  return [Math.min(...years), Math.max(...years)] as [number, number];
}

function selectedDimensionYearBounds(
  source: SourceCatalog | undefined,
  dimensions: Record<string, string[]>
) {
  if (!source) return undefined;
  const bounds = sourceDimensionEntries(source).flatMap(([key]) =>
    (dimensions[key] ?? [])
      .map(parseYearBoundsFromToken)
      .filter((item): item is [number, number] => Boolean(item))
  );
  if (bounds.length === 0) return undefined;
  return [
    Math.min(...bounds.map((item) => item[0])),
    Math.max(...bounds.map((item) => item[1]))
  ] as [number, number];
}

function sourceTemporalYearBounds(source: SourceCatalog | undefined) {
  const temporal = source?.temporal;
  if (!temporal) return undefined;
  const explicitYears = temporal.temporal_layers?.years ?? [];
  if (explicitYears.length > 0) {
    return [Math.min(...explicitYears), Math.max(...explicitYears)] as [number, number];
  }
  return temporal.available_years ?? temporal.default_years;
}

function effectiveAggregationYearBounds(
  source: SourceCatalog | undefined,
  dimensions: Record<string, string[]>
) {
  const sourceBounds = sourceTemporalYearBounds(source);
  const dimensionBounds = selectedDimensionYearBounds(source, dimensions);
  if (sourceBounds && dimensionBounds) {
    const start = Math.max(sourceBounds[0], dimensionBounds[0]);
    const end = Math.min(sourceBounds[1], dimensionBounds[1]);
    return start <= end ? [start, end] as [number, number] : undefined;
  }
  return dimensionBounds ?? sourceBounds;
}

function clampYearRangeToBounds(
  years: [number, number] | undefined,
  bounds?: [number, number]
): [number, number] | undefined {
  if (!years || !bounds) return years;
  const start = clamp(years[0], bounds[0], bounds[1]);
  const end = clamp(years[1], bounds[0], bounds[1]);
  return start <= end ? [start, end] : [end, start];
}

function dimensionsAreComplete(source: SourceCatalog | undefined, dimensions: Record<string, string[]>) {
  if (!source) return false;
  return sourceDimensionEntries(source).every(([key]) => (dimensions[key] ?? []).length > 0);
}

function sourceProviderGroups(catalog: WorkbenchCatalog) {
  const groupMeta = new Map((catalog.source_groups ?? []).map((group) => [group.id, group]));
  const groupOrder = new Map((catalog.source_groups ?? []).map((group, index) => [group.id, index]));
  const groups = new Map<string, SourceCatalog[]>();
  for (const source of catalog.sources) {
    const key = source.provider ?? "other";
    groups.set(key, [...(groups.get(key) ?? []), source]);
  }
  return [...groups.entries()]
    .sort(([left], [right]) => {
      const leftOrder = groupOrder.get(left) ?? Number.MAX_SAFE_INTEGER;
      const rightOrder = groupOrder.get(right) ?? Number.MAX_SAFE_INTEGER;
      return leftOrder - rightOrder || left.localeCompare(right);
    })
    .map(([provider, sources]) => ({
      provider,
      meta: groupMeta.get(provider),
      sources: sources.sort((a, b) => sourceDisplayName(a).localeCompare(sourceDisplayName(b)))
    }));
}

function supportedTemporalModes(source: SourceCatalog | undefined): FeatureTemporalMode[] {
  const modes = (source?.temporal?.output_modes ?? []) as FeatureTemporalMode[];
  if (!source?.temporal) return ["static"];
  return modes.length > 0 ? modes : ["static"];
}

function defaultTemporalModeForSource(source: SourceCatalog | undefined): FeatureTemporalMode {
  if (!source?.temporal) return "static";
  const modes = supportedTemporalModes(source);
  const preferred = source.temporal.default_output_mode as FeatureTemporalMode | undefined;
  if (preferred && modes.includes(preferred)) return preferred;
  for (const mode of ["supplied_layers", "aggregate", "postprocess_aggregate", "raw_slices", "static"] as FeatureTemporalMode[]) {
    if (modes.includes(mode)) return mode;
  }
  return "static";
}

function initialTemporalLayers(source: SourceCatalog | undefined): TemporalSelection["layers"] {
  const layers = source?.temporal?.temporal_layers;
  return {
    annual: false,
    annual_index: false,
    months: [],
    seasons: [],
    years: layers?.years?.length === 1 ? [layers.years[0]] : []
  };
}

function selectedTemporalLayerTokens(layers: TemporalSelection["layers"]) {
  const tokens: Array<{ key: string; label: string; query: Record<string, unknown> }> = [];
  if (layers.annual) tokens.push({ key: "annual", label: "annual", query: { period: "annual" } });
  if (layers.annual_index) tokens.push({ key: "annual_index", label: "annual index", query: {} });
  for (const month of layers.months) tokens.push({ key: month, label: month, query: { period: month } });
  for (const season of layers.seasons) tokens.push({ key: season, label: season, query: { period: season } });
  return tokens;
}

function temporalSelectionIsComplete(
  source: SourceCatalog | undefined,
  mode: FeatureTemporalMode,
  layers: TemporalSelection["layers"],
  aggregationName: string,
  aggregation?: CustomAggregation,
  aggregations: CustomAggregation[] = [],
  dimensions: Record<string, string[]> = {}
) {
  if (!source?.temporal || mode === "static") {
    return !source?.temporal || supportedTemporalModes(source).includes("static");
  }
  if (mode === "supplied_layers") {
    const hasYear = (layers.years ?? []).length > 0;
    const hasLayerToken = selectedTemporalLayerTokens(layers).length > 0;
    return hasYear || hasLayerToken;
  }
  if (mode === "aggregate" || mode === "postprocess_aggregate") {
    return aggregations.length > 0
      && aggregations.every((item) => aggregationDraftIsComplete(source, dimensions, item));
  }
  if (mode === "raw_slices") {
    const months = aggregation?.months;
    const hasMonths = Array.isArray(months) && months.length === 2
      && months[0] >= 1
      && months[1] <= 12
      && months[0] <= months[1];
    if (!hasMonths) return false;
    if (source.temporal.kind === "year_month_series") {
      const years = aggregation?.years;
      return Array.isArray(years) && years.length === 2 && years[0] <= years[1];
    }
    return true;
  }
  return true;
}

function aggregationDraftIsComplete(
  source: SourceCatalog | undefined,
  dimensions: Record<string, string[]>,
  aggregation: CustomAggregation
) {
  if (aggregation.name.trim().length === 0) return false;
  const months = aggregation.months;
  if (months && (months[0] < 1 || months[1] > 12 || months[0] > months[1])) return false;
  const years = aggregation.years;
  if (years) {
    if (years[0] > years[1]) return false;
    const bounds = effectiveAggregationYearBounds(source, dimensions);
    if (bounds && (years[0] < bounds[0] || years[1] > bounds[1])) return false;
  }
  return true;
}

function categoryFractionConfig(variable: VariableCatalog, classItem: { value?: string | number; values?: Array<string | number>; name?: string; label?: string }) {
  const classValues = categoryClassValues(classItem);
  const classToken = categoryClassToken(classItem);
  return {
    variable: variable.generated_from || variable.name.replace(/_\d{4}$/, ""),
    name: `${sanitizeToken(variable.generated_from || variable.name.replace(/_\d{4}$/, ""))}_${classToken}_fraction`,
    class_values: classValues,
    label: classItem.label || classItem.name || classToken
  };
}

function defaultResamplingForOutput(variable: VariableCatalog | undefined, isCategoryFraction = false) {
  if (isCategoryFraction) return "average";
  return variable?.resampling ?? "nearest";
}

function resamplingDescription(method: string, isCategoryFraction = false) {
  if (isCategoryFraction) {
    return method === "average"
      ? "Recommended: converts native 0/1 class membership into target-cell coverage fractions."
      : "Advanced: changes how the native category mask is transferred to the target grid.";
  }
  const descriptions: Record<string, string> = {
    nearest: "Keeps original codes/classes; safest for categorical rasters.",
    mode: "Uses the most frequent native class in each target cell.",
    average: "Averages source values; good for continuous data and binary/fraction coverage.",
    bilinear: "Smooth interpolation for continuous rasters.",
    cubic: "Smooth interpolation for continuous rasters, with more local smoothing.",
    sum: "Adds source values; use only for extensive quantities."
  };
  return descriptions[method] ?? "Advanced resampling method supported by Rasterio/GDAL.";
}

function ResamplingMethodControl({
  catalog,
  value,
  onChange,
  variable,
  isCategoryFraction = false
}: {
  catalog: WorkbenchCatalog;
  value: string;
  onChange: (value: string) => void;
  variable?: VariableCatalog;
  isCategoryFraction?: boolean;
}) {
  const methods = catalog.supported_resampling.length > 0
    ? catalog.supported_resampling
    : ["nearest", "average", "bilinear", "mode"];
  const effectiveValue = methods.includes(value) ? value : methods[0];

  return (
    <label className="resampling-control">
      Output resampling
      <select value={effectiveValue} onChange={(event) => onChange(event.target.value)}>
        {methods.map((method) => (
          <option key={method} value={method}>{method}</option>
        ))}
      </select>
      <small className="field-hint">
        {resamplingDescription(effectiveValue, isCategoryFraction)}
        {variable?.resampling && !isCategoryFraction ? ` Source default: ${variable.resampling}.` : ""}
      </small>
    </label>
  );
}

function aggregationDefaults(source: SourceCatalog | undefined, variables: string[], metric = "mean"): CustomAggregation {
  const temporal = source?.temporal;
  const availableYears = temporal?.temporal_layers?.years ?? [];
  const years = temporal?.default_years
    ?? temporal?.available_years
    ?? (availableYears.length > 0 ? [availableYears[0], availableYears[availableYears.length - 1]] as [number, number] : undefined);
  const form = temporal?.aggregation_forms?.[0] ?? (years ? "year_range_metric" : "month_range_metric");
  return {
    name: "period_mean",
    form,
    metric,
    months: form.includes("month") || temporal?.default_months ? defaultRange(temporal?.default_months) : undefined,
    years,
    variables
  };
}

function allValuesSelected<T extends string | number>(selected: T[], values: T[]) {
  return values.length > 0 && selected.length === values.length && values.every((value) => selected.includes(value));
}

function sourceTemporalSelection(source: SourceCatalog): TemporalSelection {
  const temporal = source.temporal;

  return {
    outputMode: temporal?.default_output_mode ?? "static",
    months: defaultRange(temporal?.default_months),
    years: temporal?.default_years ? defaultRange(temporal.default_years) : undefined,
    layers: {
      annual: false,
      annual_index: false,
      months: [],
      seasons: [],
      years: []
    },
    aggregationUse: [],
    customAggregations: []
  };
}

function cartesianProduct<T>(groups: T[][]): T[][] {
  if (groups.length === 0) return [[]];
  return groups.reduce<T[][]>(
    (acc, group) => acc.flatMap((prefix) => group.map((value) => [...prefix, value])),
    [[]]
  );
}

function selectedDimensionContexts(source: SourceCatalog, dimensions: Record<string, string[]>) {
  const entries = sourceDimensionEntries(source);
  if (entries.length === 0) return [{ dimensions: {}, query: {}, suffix: "" }];
  const valueGroups = entries.map(([key, values]) => {
    const selected = dimensions[key] && dimensions[key].length > 0 ? dimensions[key] : values.slice(0, 1);
    return selected.map((value) => ({ key, value }));
  });

  return cartesianProduct(valueGroups).map((items) => {
    const selectedDimensions: Record<string, string[]> = {};
    const query: Record<string, string> = {};
    const suffix: string[] = [];
    for (const item of items) {
      selectedDimensions[item.key] = [item.value];
      const contextKey = source.dimension_context_keys?.[item.key] ?? item.key;
      query[contextKey] = item.value;
      suffix.push(item.value);
    }
    return { dimensions: selectedDimensions, query, suffix: suffix.join("_") };
  });
}

function applyOutputVariablePattern(variable: VariableCatalog, replacements: Record<string, string | number>) {
  const pattern = typeof variable.temporal?.variable_pattern === "string"
    ? variable.temporal.variable_pattern
    : variable.name;
  return Object.entries(replacements).reduce(
    (text, [key, value]) => text.split(`{${key}}`).join(String(value)),
    pattern
  );
}

function sourceQueryVariableForContext(
  source: SourceCatalog,
  variable: VariableCatalog,
  context: { suffix: string },
  categoryFraction?: CategoryFractionSelection
) {
  if (categoryFraction) return categoryFraction.name;
  if (source.temporal?.kind === "yearly_static_collection" && context.suffix) {
    return `${variable.name}_${sanitizeToken(context.suffix)}`;
  }
  return variable.kind === "vector_layer"
    ? variable.name.split(".").slice(-1)[0]
    : variable.name;
}

function rangeValues(range?: [number, number]) {
  if (!range) return [];
  const [start, end] = range;
  const values: number[] = [];
  for (let value = start; value <= end; value += 1) values.push(value);
  return values;
}

function sourceInputForOutput({
  source,
  variable,
  query,
  dimensions,
  temporal,
  categoryFraction,
  sourceResolution,
  resampling
}: {
  source: SourceCatalog;
  variable: VariableCatalog;
  query: DerivedInputQuery;
  dimensions?: Record<string, string[]>;
  temporal?: Record<string, unknown>;
  categoryFraction?: CategoryFractionSelection;
  sourceResolution?: string;
  resampling?: string;
}): FeatureSourceInput {
  const isLayer = variable.kind === "vector_layer";
  return {
    kind: "source",
    source_id: source.id,
    config: source.config_path,
    variable: isLayer ? undefined : variable.name,
    layer: isLayer ? variable.name : undefined,
    query,
    dimensions,
    temporal,
    category_fraction: categoryFraction,
    source_resolution: sourceResolution,
    resampling
  };
}

function buildSourceLayerOutputs({
  featureName,
  source,
  variable,
  dimensions,
  temporalMode,
  years,
  layers,
  aggregation,
  categoryFraction,
  sourceResolution,
  resampling
}: {
  featureName: string;
  source: SourceCatalog;
  variable: VariableCatalog;
  dimensions: Record<string, string[]>;
  temporalMode: FeatureTemporalMode;
  years: number[];
  layers?: TemporalSelection["layers"];
  aggregation?: CustomAggregation;
  categoryFraction?: CategoryFractionSelection;
  sourceResolution?: string;
  resampling?: string;
}): DatasetFeatureOutput[] {
  const contexts = selectedDimensionContexts(source, dimensions);
  const outputs: DatasetFeatureOutput[] = [];
  const selectedLayers = layers ?? {
    annual: false,
    annual_index: false,
    months: [],
    seasons: [],
    years
  };
  const temporalLayers = {
    annual: selectedLayers.annual,
    annual_index: selectedLayers.annual_index,
    months: selectedLayers.months,
    seasons: selectedLayers.seasons,
    years: years.length > 0 ? years : selectedLayers.years
  };

  for (const context of contexts) {
    if (temporalMode === "supplied_layers") {
      const selectedYears = years.length > 0 ? years : temporalLayers.years;
      for (const year of selectedYears) {
        const variableName = applyOutputVariablePattern(variable, { ...context.query, year });
        const categoryName = categoryFraction
          ? selectedYears.length > 1 ? `${categoryFraction.name}_${year}` : categoryFraction.name
          : undefined;
        const suffix = [context.suffix, year].filter(Boolean).join("_");
        const temporal = {
          output_mode: "supplied_layers",
          layers: temporalLayers
        };
        const query = {
          source_id: source.id,
          variable: categoryName ?? variableName,
          ...context.query
        };
        outputs.push({
          name: suffix ? `${featureName}_${sanitizeToken(suffix)}` : sanitizeToken(featureName),
          suffix,
          temporal_key: String(year),
          dimension_key: context.suffix,
          source: sourceInputForOutput({
            source,
            variable,
            query,
            dimensions: context.dimensions,
            temporal,
            categoryFraction,
            sourceResolution,
            resampling
          })
        });
      }

      for (const token of selectedTemporalLayerTokens(selectedLayers)) {
        const suffix = [context.suffix, token.key].filter(Boolean).join("_");
        const temporal = {
          output_mode: "supplied_layers",
          layers: temporalLayers
        };
        const query = {
          source_id: source.id,
          variable: categoryFraction?.name ?? variable.name,
          ...context.query,
          ...token.query
        };
        outputs.push({
          name: suffix ? `${featureName}_${sanitizeToken(suffix)}` : sanitizeToken(featureName),
          suffix,
          temporal_key: token.key,
          dimension_key: context.suffix,
          source: sourceInputForOutput({
            source,
            variable,
            query,
            dimensions: context.dimensions,
            temporal,
            categoryFraction,
            sourceResolution,
            resampling
          })
        });
      }
      continue;
    }

    if (temporalMode === "raw_slices" && aggregation) {
      const months = rangeValues(aggregation.months ?? [1, 12]);
      const rawYears = source.temporal?.kind === "year_month_series"
        ? rangeValues(aggregation.years)
        : [undefined];
      const temporal = {
        output_mode: "raw_slices",
        months: aggregation.months,
        years: source.temporal?.kind === "year_month_series" ? aggregation.years : undefined
      };
      for (const rawYear of rawYears) {
        for (const month of months) {
          const monthToken = `${month}`.padStart(2, "0");
          const aggregationName = rawYear ? `raw_${rawYear}_${monthToken}` : `month_${monthToken}`;
          const suffix = [context.suffix, rawYear, `m${monthToken}`].filter(Boolean).join("_");
          const query = {
            source_id: source.id,
            variable: sourceQueryVariableForContext(source, variable, context, categoryFraction),
            aggregation_name: aggregationName,
            months: [month],
            ...context.query
          };
          outputs.push({
            name: suffix ? `${featureName}_${sanitizeToken(suffix)}` : sanitizeToken(featureName),
            suffix,
            temporal_key: rawYear ? `${rawYear}-${monthToken}` : `m${monthToken}`,
            dimension_key: context.suffix,
            source: sourceInputForOutput({
              source,
              variable,
              query,
              dimensions: context.dimensions,
              temporal,
              categoryFraction,
              sourceResolution,
              resampling
            })
          });
        }
      }
      continue;
    }

    if ((temporalMode === "aggregate" || temporalMode === "postprocess_aggregate") && aggregation) {
      const suffix = [context.suffix, aggregation.name].filter(Boolean).join("_");
      const temporal = {
        output_mode: temporalMode,
        aggregations: {
          use: [],
          custom: [aggregation]
        }
      };
      const query = {
        source_id: source.id,
        variable: sourceQueryVariableForContext(source, variable, context, categoryFraction),
        aggregation_name: aggregation.name,
        ...context.query
      };
      outputs.push({
        name: suffix ? `${featureName}_${sanitizeToken(suffix)}` : sanitizeToken(featureName),
        suffix,
        temporal_key: aggregation.name,
        dimension_key: context.suffix,
        source: sourceInputForOutput({
          source,
          variable,
          query,
          dimensions: context.dimensions,
          temporal,
          categoryFraction,
          sourceResolution,
          resampling
        })
      });
      continue;
    }

    const suffix = context.suffix;
    const query = {
      source_id: source.id,
      variable: sourceQueryVariableForContext(source, variable, context, categoryFraction),
      ...context.query
    };
    outputs.push({
      name: suffix ? `${featureName}_${sanitizeToken(suffix)}` : sanitizeToken(featureName),
      suffix,
      dimension_key: context.suffix,
      source: sourceInputForOutput({
        source,
        variable,
        query,
        dimensions: context.dimensions,
        temporal: source.temporal ? { output_mode: "static" } : undefined,
        categoryFraction,
        sourceResolution,
        resampling
      })
    });
  }

  return outputs;
}

function dimensionPatternContexts(source: SourceCatalog, selection: SourceSelection) {
  const entries = Object.entries(source.dimension_context_keys ?? {});
  if (entries.length === 0) return [{} as Record<string, string>];

  let contexts: Array<Record<string, string>> = [{}];
  for (const [dimensionKey, contextKey] of entries) {
    const values = selection.dimensions[dimensionKey] ?? [];
    if (values.length === 0) return [];
    contexts = contexts.flatMap((context) =>
      values.map((value) => ({
        ...context,
        [contextKey]: value
      }))
    );
  }
  return contexts;
}

function applyVariablePattern(pattern: string, replacements: Record<string, string | number>) {
  return Object.entries(replacements).reduce(
    (text, [key, value]) => text.split(`{${key}}`).join(String(value)),
    pattern
  );
}

function toggleValue<T extends string | number>(values: T[], value: T) {
  return values.includes(value)
    ? values.filter((item) => item !== value)
    : [...values, value];
}

function toggleStage(values: string[], stage: string) {
  if (stage === "all") {
    return values.includes("all") ? [] : ["all"];
  }
  return toggleValue(values.filter((item) => item !== "all"), stage);
}

function defaultStages(supportedStages: string[] = ["download", "clip", "build", "all"]) {
  if (supportedStages.includes("all")) return ["all"];
  return [supportedStages[0] ?? "build"];
}

function buildTemporalConfig(source: SourceCatalog | undefined, selection: SourceSelection) {
  const capability = source?.temporal;
  const temporal = selection.temporal;
  const mode = temporal.outputMode;

  if (!capability || mode === "static") {
    return undefined;
  }

  if (mode === "aggregate") {
    const custom = temporal.customAggregations.map((item) => {
      const base: Record<string, unknown> = {
        name: item.name,
        form: item.form,
        variables: item.variables
      };
      if (item.months) base.months = item.months;
      if (item.years) base.years = item.years;
      if (item.form === "year_then_across_years") {
        base.within_year_metric = item.within_year_metric ?? "sum";
        base.across_year_metric = item.across_year_metric ?? "mean";
        if (item.output_metric_name) base.output_metric_name = item.output_metric_name;
      } else {
        base.metric = item.metric;
      }
      if (item.threshold !== undefined) base.threshold = item.threshold;
      if (item.comparison) base.comparison = item.comparison;
      return base;
    });

    return {
      output_mode: "aggregate",
      aggregations: {
        use: temporal.aggregationUse,
        custom
      }
    };
  }

  if (mode === "raw_slices") {
    return {
      output_mode: "raw_slices",
      months: temporal.months,
      years: capability.kind === "year_month_series" ? temporal.years : undefined
    };
  }

  if (mode === "supplied_layers") {
    return {
      output_mode: "supplied_layers",
      layers: {
        annual: temporal.layers.annual,
        annual_index: temporal.layers.annual_index,
        months: temporal.layers.months,
        seasons: temporal.layers.seasons,
        years: temporal.layers.years
      }
    };
  }

  if (mode === "postprocess_aggregate") {
    const custom = temporal.customAggregations.map((item) => {
      const base: Record<string, unknown> = {
        name: item.name,
        form: item.form,
        variables: item.variables,
        metric: item.metric
      };
      if (item.months) base.months = item.months;
      if (item.years) base.years = item.years;
      if (item.threshold !== undefined) base.threshold = item.threshold;
      if (item.comparison) base.comparison = item.comparison;
      if (item.start_date) base.start_date = item.start_date;
      if (item.end_date) base.end_date = item.end_date;
      return base;
    });

    return {
      output_mode: "postprocess_aggregate",
      aggregations: {
        use: temporal.aggregationUse,
        custom
      }
    };
  }

  return undefined;
}

function createSelection(source: SourceCatalog): SourceSelection {
  const variables = sourceVariables(source)
    .filter((item) => item.enabled_default)
    .map((item) => item.name);

  const dimensions: Record<string, string[]> = {};
  for (const [key] of sourceDimensionEntries(source)) {
    dimensions[key] = [];
  }

  return {
    id: source.id,
    config: source.config_path,
    selected: false,
    stages: [],
    sourceResolution: source.source_resolution,
    keepRawAfterClip: source.keep_raw_after_clip_default ?? true,
    variables,
    layers: [],
    categoryFractions: [],
    dimensions,
    temporal: sourceTemporalSelection(source),
    resamplingByVariable: {}
  };
}

function App() {
  const [catalog, setCatalog] = useState<WorkbenchCatalog | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("Project");
  const [activeSourceId, setActiveSourceId] = useState<string>("");
  const [runName, setRunName] = useState("pallars_workbench_100m");
  const [description, setDescription] = useState("Workbench-generated Pirineus Raster dataset.");
  const [projectConfig, setProjectConfig] = useState("configs/project.yaml");
  const [targetCrs, setTargetCrs] = useState("EPSG:3035");
  const [aoiPath, setAoiPath] = useState("configs/aoi/experimental_pallars_sobira.yaml");
  const [resolution, setResolution] = useState(100);
  const [stages, setStages] = useState<string[]>(["all"]);
  const [datasetDir, setDatasetDir] = useState("data_processed/datasets/pallars_workbench_100m");
  const [createdAois, setCreatedAois] = useState<AoiCatalog[]>([]);
  const [selections, setSelections] = useState<Record<string, SourceSelection>>({});
  const [derivedFeatures, setDerivedFeatures] = useState<DerivedFeatureConfig[]>([]);
  const [datasetFeatures, setDatasetFeatures] = useState<DatasetFeatureConfig[]>([]);
  const [projectStep, setProjectStep] = useState<"setup" | "features" | "review">("setup");
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [serverYaml, setServerYaml] = useState("");
  const [apiError, setApiError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [hasStarted, setHasStarted] = useState(false);
  const [startMode, setStartMode] = useState<StartMode>("menu");
  const [backgroundUrls, setBackgroundUrls] = useState<string[]>([]);
  const [backgroundUrl, setBackgroundUrl] = useState<string | null>(null);

  useEffect(() => {
    fetchCatalog()
      .then((data) => {
        setCatalog(data);
        setCatalogError(null);
        setProjectConfig(data.project.config_path);
        setTargetCrs(data.project.crs ?? "EPSG:3035");
        setResolution(data.project.default_resolution_m ?? data.project.available_resolutions_m[0] ?? 100);
        setStages(defaultStages(data.supported_stages));
        if (data.aois[0]) {
          setAoiPath(data.aois[0].path);
        }

        const nextSelections: Record<string, SourceSelection> = {};
        for (const source of data.sources) {
          nextSelections[source.id] = createSelection(source);
        }
        setSelections(nextSelections);
        setActiveSourceId(data.sources[0]?.id ?? "");
      })
      .catch((error: Error) => {
        setCatalogError(error.message);
      })
      .finally(() => {
        setCatalogLoading(false);
      });
  }, []);

  useEffect(() => {
    let cancelled = false;

    fetch(backgroundManifestUrl, { cache: "no-cache" })
      .then((response) => response.ok ? response.json() : null)
      .then((manifest: unknown) => {
        if (cancelled) return;
        const images = normalizeBackgroundUrls(
          manifest && typeof manifest === "object" && "images" in manifest
            ? (manifest as { images?: unknown }).images
            : undefined
        );
        setBackgroundUrls(images);
        const nextBackground = pickRandomBackground(images);
        if (!nextBackground) return;
        preloadBackground(nextBackground)
          .then((loadedUrl) => {
            if (!cancelled) setBackgroundUrl(loadedUrl);
          })
          .catch(() => undefined);
      })
      .catch(() => {
        if (cancelled) return;
        setBackgroundUrls([]);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (backgroundUrls.length <= 1 || !backgroundUrl) return undefined;

    const intervalId = window.setInterval(() => {
      const nextBackground = pickRandomBackground(backgroundUrls, backgroundUrl);
      if (!nextBackground) return;
      preloadBackground(nextBackground)
        .then((loadedUrl) => setBackgroundUrl(loadedUrl))
        .catch(() => undefined);
    }, backgroundRotationMs);

    return () => window.clearInterval(intervalId);
  }, [backgroundUrl, backgroundUrls]);

  const selectedSources = useMemo(
    () => Object.values(selections).filter((selection) => selection.selected),
    [selections]
  );

  const selectedCatalogSources = useMemo(
    () => catalog?.sources.filter((source) => selections[source.id]?.selected) ?? [],
    [catalog?.sources, selections]
  );

  const plannedReviewLayers = useMemo(
    () => catalog ? buildPlannedLayers(catalog, selectedSources) : [],
    [catalog, selectedSources]
  );

  const availableAois = useMemo(() => {
    const byPath = new Map<string, AoiCatalog>();
    for (const aoi of [...(catalog?.aois ?? []), ...createdAois]) {
      byPath.set(aoi.path, aoi);
    }
    return [...byPath.values()];
  }, [catalog?.aois, createdAois]);

  const activeSource = useMemo(
    () => selectedCatalogSources.find((source) => source.id === activeSourceId),
    [selectedCatalogSources, activeSourceId]
  );

  useEffect(() => {
    if (!catalog || selectedCatalogSources.length === 0) return;
    if (!selectedCatalogSources.some((source) => source.id === activeSourceId)) {
      setActiveSourceId(selectedCatalogSources[0].id);
    }
  }, [activeSourceId, catalog, selectedCatalogSources]);

  const runConfig = useMemo(() => {
    return {
      run: {
        name: runName,
        description,
        project_config: projectConfig,
        crs: targetCrs,
        aoi_config: aoiPath,
        resolution_m: resolution,
        stages
      },
      features: datasetFeatures,
      outputs: {
        dataset_dir: datasetDir,
        copy_rasters: true,
        overwrite_existing: true,
        write_run_summary: true,
        write_manifest: true
      }
    };
  }, [
    aoiPath,
    catalog?.sources,
    datasetDir,
    description,
    projectConfig,
    targetCrs,
    resolution,
    runName,
    selectedSources,
    stages,
    datasetFeatures
  ]);

  const localYaml = useMemo(() => renderYaml(runConfig), [runConfig]);
  const yamlText = serverYaml || localYaml;
  const apiStatus = catalogError ? "API offline" : catalog ? "API ready" : "Loading API";

  function patchSelection(sourceId: string, patch: Partial<SourceSelection>) {
    setSelections((current) => ({
      ...current,
      [sourceId]: {
        ...current[sourceId],
        ...patch
      }
    }));
  }

  function patchSelectionMut(sourceId: string, edit: (selection: SourceSelection) => SourceSelection) {
    setSelections((current) => ({
      ...current,
      [sourceId]: edit(current[sourceId])
    }));
  }

  async function validate() {
    setApiError(null);
    try {
      const report = await validateRunConfig(runConfig);
      setValidation(report);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : String(error));
    }
  }

  async function renderFromServer() {
    setApiError(null);
    setSaveStatus(null);
    try {
      const result = await renderRunConfig(runConfig);
      setValidation(result.validation);
      setServerYaml(result.yaml);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : String(error));
    }
  }

  async function copyYaml() {
    await navigator.clipboard.writeText(yamlText);
  }

  async function saveYamlToRuns() {
    setApiError(null);
    setSaveStatus(null);
    try {
      const result = await saveRunConfig(runConfig, runName);
      setValidation(result.validation);
      setServerYaml(result.yaml);
      setSaveStatus(`Saved as: ${result.path}`);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : String(error));
    }
  }

  function downloadYaml() {
    const blob = new Blob([yamlText], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${runName}.yaml`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function removePlannedLayer(layer: PlannedLayer) {
    const dependents = derivedFeatures
      .filter((feature) => derivedFeatureDependsOnLayer(feature, layer))
      .map((feature) => feature.name);
    if (dependents.length > 0) {
      const shouldRemove = window.confirm(
        `Removing ${layer.label} will also remove these derived layers: ${dependents.join(", ")}. Continue?`
      );
      if (!shouldRemove) return;
    }

    setDerivedFeatures((current) =>
      current.filter((feature) => !derivedFeatureDependsOnLayer(feature, layer))
    );
    setSelections((current) => {
      const sourceId = layer.query.source_id;
      if (!sourceId || !current[sourceId]) return current;
      const isCategoryFraction = current[sourceId].categoryFractions.some(
        (item) => item.name === layer.variable
      );
      if (isCategoryFraction) {
        const nextResampling = { ...current[sourceId].resamplingByVariable };
        delete nextResampling[layer.variable];
        return {
          ...current,
          [sourceId]: {
            ...current[sourceId],
            categoryFractions: current[sourceId].categoryFractions.filter(
              (item) => item.name !== layer.variable
            ),
            resamplingByVariable: nextResampling
          }
        };
      }
      const variableName = layer.baseVariable ?? layer.variable;
      return {
        ...current,
        [sourceId]: {
          ...current[sourceId],
          variables: current[sourceId].variables.filter((name) => name !== variableName)
        }
      };
    });
  }

  if (!backgroundUrl) {
    return (
      <main className="background-loading-screen">
        <div>
          <span className="eyebrow">Pirineus Raster</span>
          <h1>Loading background</h1>
        </div>
      </main>
    );
  }

  if (!hasStarted) {
    return (
      <main className="welcome-screen">
        <BackgroundImage src={backgroundUrl} variant="welcome" />
        <section className="welcome-hero">
          <div className="home-kicker">Environmental raster workbench</div>
          <h1>Welcome to Pirineus Raster</h1>
          <p>
            Build clean, reproducible raster datasets for Pyrenean research from
            climate, terrain, land-cover and geology sources.
          </p>
          <button className="home-cta" onClick={() => setHasStarted(true)}>
            Start building my personalized dataset
          </button>
        </section>
        <BackgroundCredit />
      </main>
    );
  }

  if (startMode === "menu") {
    return (
      <WorkbenchShell backgroundUrl={backgroundUrl}>
        <header className="topbar">
          <div>
            <h1>Pirineus Raster</h1>
            <p>{targetCrs} · choose how you want to start</p>
          </div>
          <div className={`api-pill ${catalogError ? "bad" : catalog ? "good" : "loading"}`}>
            {apiStatus}
          </div>
        </header>
        <StartModePanel setStartMode={setStartMode} />
        <BackgroundCredit />
      </WorkbenchShell>
    );
  }

  if (startMode === "aoi") {
    return (
      <WorkbenchShell backgroundUrl={backgroundUrl}>
        <header className="topbar">
          <div>
            <h1>New AOI</h1>
            <p>Create an AOI config and target grid for this project.</p>
          </div>
          <button className="ghost" onClick={() => setStartMode("menu")}>Back</button>
        </header>
        <AoiBuilderPanel
          projectConfig={projectConfig}
          resolutions={catalog?.project.available_resolutions_m ?? [resolution]}
          targetCrs={targetCrs}
          setTargetCrs={setTargetCrs}
          aoiPath={aoiPath}
          setAoiPath={setAoiPath}
          resolution={resolution}
          setResolution={setResolution}
          onAoiCreated={(aoi) => {
            setCreatedAois((current) => [...current.filter((item) => item.path !== aoi.path), aoi]);
            setAoiPath(aoi.path);
          }}
        />
        <BackgroundCredit />
      </WorkbenchShell>
    );
  }

  if (startMode === "sources") {
    return (
      <WorkbenchShell backgroundUrl={backgroundUrl}>
        <header className="topbar">
          <div>
            <h1>Sources Information</h1>
            <p>{catalog?.sources.length ?? 0} source configs available locally.</p>
          </div>
          <button className="ghost" onClick={() => setStartMode("menu")}>Back</button>
        </header>
        {catalog ? <SourcesInfoPanel catalog={catalog} /> : (
          <section className={`notice ${catalogError ? "error" : "info"}`}>
            {catalogLoading ? "Loading project catalog..." : `Config API is not available: ${catalogError}`}
          </section>
        )}
        <BackgroundCredit />
      </WorkbenchShell>
    );
  }

  if (startMode === "tutorial") {
    return (
      <WorkbenchShell backgroundUrl={backgroundUrl}>
        <header className="topbar">
          <div>
            <h1>Workbench Guide</h1>
            <p>How to build final features, choose resampling and avoid common traps.</p>
          </div>
          <button className="ghost" onClick={() => setStartMode("menu")}>Back</button>
        </header>
        <TutorialPanel />
        <BackgroundCredit />
      </WorkbenchShell>
    );
  }

  if (startMode === "project") {
    const stepTitle = projectStep === "setup"
      ? "Project Setup"
      : projectStep === "features"
        ? "Feature Builder"
        : "Review";

    return (
      <WorkbenchShell backgroundUrl={backgroundUrl}>
        <header className="topbar">
          <div>
            <h1>{stepTitle}</h1>
            <p>{targetCrs} · {datasetFeatures.length} final features · {validation?.estimated_layers ?? 0} estimated layers</p>
          </div>
          <div className="topbar-actions">
            <button className="ghost" onClick={() => setStartMode("menu")}>Home</button>
            {projectStep !== "setup" && (
              <button className="ghost" onClick={() => setProjectStep(projectStep === "review" ? "features" : "setup")}>
                Back
              </button>
            )}
            <div className={`api-pill ${catalogError ? "bad" : catalog ? "good" : "loading"}`}>
              {apiStatus}
            </div>
          </div>
        </header>

        <nav className="feature-stepper" aria-label="Project workflow">
          {(["setup", "features", "review"] as const).map((step) => (
            <button
              key={step}
              className={projectStep === step ? "active" : ""}
              disabled={step === "features" && !catalog}
              onClick={() => {
                if (step === "review" && datasetFeatures.length === 0) return;
                setProjectStep(step);
              }}
            >
              {step === "setup" ? "Project setup" : step === "features" ? "Final features" : "Review"}
            </button>
          ))}
        </nav>

        {!catalog && (
          <section className={`notice ${catalogError ? "error" : "info"}`}>
            {catalogLoading
              ? "Loading project catalog from the local config API..."
              : `Config API is not available: ${catalogError}. Start it with "python3 -m src.cli.main serve-config-api --host 127.0.0.1 --port 8765" from the repository root.`}
          </section>
        )}

        {projectStep === "setup" && catalog && (
          <FeatureProjectSetupPanel
            catalog={catalog}
            aois={availableAois}
            runName={runName}
            setRunName={setRunName}
            description={description}
            setDescription={setDescription}
            projectConfig={projectConfig}
            setProjectConfig={setProjectConfig}
            targetCrs={targetCrs}
            setTargetCrs={setTargetCrs}
            aoiPath={aoiPath}
            setAoiPath={setAoiPath}
            resolution={resolution}
            setResolution={setResolution}
            stages={stages}
            setStages={setStages}
            datasetDir={datasetDir}
            setDatasetDir={setDatasetDir}
            onNext={() => setProjectStep("features")}
          />
        )}

        {projectStep === "features" && catalog && (
          <FeatureBuilderPanel
            catalog={catalog}
            features={datasetFeatures}
            setFeatures={setDatasetFeatures}
            projectName={runName}
            onReview={() => setProjectStep("review")}
          />
        )}

        {projectStep === "review" && (
          <FeatureReviewPanel
            yamlText={yamlText}
            validation={validation}
            apiError={apiError}
            saveStatus={saveStatus}
            validate={validate}
            renderFromServer={renderFromServer}
            copyYaml={copyYaml}
            saveYamlToRuns={saveYamlToRuns}
            downloadYaml={downloadYaml}
            features={datasetFeatures}
            setFeatures={setDatasetFeatures}
          />
        )}
        <BackgroundCredit />
      </WorkbenchShell>
    );
  }

  return null;
}

function StartModePanel({ setStartMode }: { setStartMode: (mode: StartMode) => void }) {
  return (
    <main className="workspace start-mode-grid">
      <button className="start-mode-card" onClick={() => setStartMode("project")}>
        <strong>Start new project</strong>
        <small>Open the dataset configuration workflow.</small>
      </button>
      <button className="start-mode-card" onClick={() => setStartMode("aoi")}>
        <strong>New AOI</strong>
        <small>Create a new area-of-interest config and grid.</small>
      </button>
      <button className="start-mode-card" onClick={() => setStartMode("sources")}>
        <strong>Sources information</strong>
        <small>Browse available sources, sub-sources and variables.</small>
      </button>
      <button className="start-mode-card tutorial-entry-card" onClick={() => setStartMode("tutorial")}>
        <strong>How the workbench works</strong>
        <small>Read the practical guide before building a dataset: final features, temporal choices, masks, category fractions, resampling and examples.</small>
      </button>
    </main>
  );
}

function TutorialPanel() {
  const [section, setSection] = useState("overview");
  const sections = [
    ["overview", "Overview"],
    ["project", "New project"],
    ["custom", "Single feature"],
    ["official", "Official layers"],
    ["masks", "Fractions and masks"],
    ["examples", "Examples"],
    ["limits", "Limits"]
  ];

  return (
    <main className="workspace tutorial-workspace">
      <section className="panel tutorial-nav">
        <h2>Workbench guide</h2>
        <p className="builder-copy">A precise map of what each tool does and when to use it.</p>
        <div className="tutorial-section-list">
          {sections.map(([value, label]) => (
            <button
              key={value}
              className={section === value ? "active" : ""}
              onClick={() => setSection(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </section>

      <section className="panel tutorial-content">
        {section === "overview" && (
          <>
            <h2>Think in final features</h2>
            <p>
              A project is built around final dataset variables. Official source rasters, temporal slices and
              intermediate layers can be downloaded and clipped during the run, but the final manifest is meant
              to contain only the features that you explicitly confirm.
            </p>
            <p>
              The safest mental model is: every card in the right sidebar should be a variable you would be happy
              to see in the final modelling table. If an official raster is only needed to compute another variable,
              use it as an input inside a custom feature rather than adding it as an official final layer.
            </p>
            <div className="tutorial-grid">
              <article>
                <strong>Project setup</strong>
                <span>Defines run name, AOI, CRS, target resolution, output directory and stages.</span>
              </article>
              <article>
                <strong>Feature builder</strong>
                <span>Creates final features from official layers, recipes, masks, spatial operations or expressions.</span>
              </article>
              <article>
                <strong>Review</strong>
                <span>Validates, renders and saves the YAML that the CLI will run.</span>
              </article>
            </div>
            <div className="tutorial-subsection">
              <h3>What happens internally</h3>
              <ul>
                <li>The YAML stores final features, not a long source-first shopping list.</li>
                <li>The compiler derives the source requirements needed to build those final features.</li>
                <li>Download and clip steps may create intermediate rasters, but those are implementation details.</li>
                <li>Derived features can use official inputs or features already created earlier in the same project.</li>
              </ul>
            </div>
          </>
        )}

        {section === "project" && (
          <>
            <h2>New project</h2>
            <p>
              Use this when you want to build a dataset. First define the project envelope: AOI, target CRS,
              target resolution and output folder. All features created later are aligned to that target grid.
            </p>
            <ul>
              <li><strong>AOI</strong> controls the spatial extent. Create it in New AOI if your study area is not already configured.</li>
              <li><strong>Target CRS</strong> is the CRS of the final rasters. Source CRS is handled internally during clip/build.</li>
              <li><strong>Resolution</strong> is the final grid size, for example 100 m. Resampling choices decide how native rasters are transferred to that grid.</li>
              <li><strong>Stages</strong> should usually stay on all; split stages only for debugging or reruns.</li>
            </ul>
            <div className="tutorial-subsection">
              <h3>Recommended order</h3>
              <ul>
                <li>Create or select the AOI before choosing variables, because it controls which source tiles matter.</li>
                <li>Choose the target CRS from the final analysis needs, not from the source CRS. The tool reprojects sources internally.</li>
                <li>Choose the target resolution from the ecological question. A 100 m dataset is good for broad habitat modelling; smaller cells increase runtime and storage.</li>
                <li>Keep the output directory tied to the run name so repeated experiments do not overwrite each other.</li>
              </ul>
            </div>
          </>
        )}

        {section === "custom" && (
          <>
            <h2>Build single feature</h2>
            <p>
              Use this when one final variable needs a deliberate construction. You first name the final feature,
              then choose one operation family.
            </p>
            <div className="tutorial-grid">
              <article>
                <strong>Use official source layer</strong>
                <span>One source variable, optionally with dimensions, temporal selection, category fraction and resampling.</span>
              </article>
              <article>
                <strong>Guided recipe</strong>
                <span>Strict formulas such as thermal range, water balance or snow persistence. Inputs are filtered to compatible variables.</span>
              </article>
              <article>
                <strong>Masking</strong>
                <span>Creates final-grid binary 0/1 rasters from thresholds or class equality.</span>
              </article>
              <article>
                <strong>Spatial operation</strong>
                <span>DEM terrain derivatives, focal windows or distances to masks/classes.</span>
              </article>
              <article>
                <strong>Advanced expression</strong>
                <span>Map algebra using x, y and z. Multi-input temporal outputs are intersected by matching temporal keys.</span>
              </article>
            </div>
            <div className="tutorial-subsection">
              <h3>Input selection rules</h3>
              <ul>
                <li>DEM terrain operations only accept DEM/elevation-like variables.</li>
                <li>Distance operations accept binary masks, categorical rasters or class-derived masks.</li>
                <li>Thermal range expects maximum and minimum temperature inputs, not arbitrary climate variables.</li>
                <li>Expression inputs y and z are optional until the expression text references them.</li>
                <li>When two temporal inputs are used together, only matching temporal outputs are combined. Non-temporal dimensions expand by product.</li>
              </ul>
            </div>
            <div className="tutorial-subsection">
              <h3>When to use features already created</h3>
              <p>
                After confirming a final feature, it appears as a project input in later selectors. This is useful
                for chained workflows such as category fraction first, focal smoothing second, and expression third.
              </p>
            </div>
          </>
        )}

        {section === "official" && (
          <>
            <h2>Add official source layers</h2>
            <p>
              Use this when you want to add several official variables without derived processing. Select the
              source product, choose variables and category fractions, complete dimensions, temporal mode and
              resampling, then add them as final features.
            </p>
            <ul>
              <li><strong>Dimensions</strong> are non-temporal variants such as GCM, SSP, period, season or product-specific axes.</li>
              <li><strong>Temporal supplied layers</strong> keeps existing years, months, seasons or index layers from the source.</li>
              <li><strong>Temporal aggregate</strong> creates a named summary over available source time steps.</li>
              <li><strong>Raw slices</strong> keeps smaller temporal units when the source supports them.</li>
            </ul>
            <div className="tutorial-subsection">
              <h3>How to avoid accidental outputs</h3>
              <ul>
                <li>Select the original variable only if you really want its raster in the final dataset.</li>
                <li>Select category fractions if you want each class as its own proportional variable.</li>
                <li>If you select fractions but not the original categorical variable, the source can still be used internally without adding the class-code raster to the final dataset.</li>
                <li>Use the final resampling/source-resolution panel to decide how the source becomes the project grid.</li>
              </ul>
            </div>
          </>
        )}

        {section === "masks" && (
          <>
            <h2>Category fractions, masks and resampling</h2>
            <p>
              Category fractions and class masks are related but not interchangeable. Use category fractions for
              coverage percentages; use class masks for final-grid binary rasters.
            </p>
            <div className="tutorial-grid">
              <article>
                <strong>Category fraction</strong>
                <span>The source categorical raster is converted to 0/1 at native resolution, then resampled. With average, the output is a 0-1 coverage ratio per target cell.</span>
              </article>
              <article>
                <strong>Class mask</strong>
                <span>The already aligned target-grid raster is tested with where(x == class). It returns 0/1 and cannot recover sub-cell composition lost during resampling.</span>
              </article>
              <article>
                <strong>Resampling</strong>
                <span>Choose average for category fractions, nearest or mode for original categorical codes, and bilinear/average for continuous rasters depending on the source.</span>
              </article>
            </div>
            <div className="notice info compact-notice">
              For habitat percentages at 100 m from a 10 m categorical map, do not build the original class and
              then mask it. Select the desired category fractions and keep average resampling.
            </div>
            <div className="tutorial-subsection">
              <h3>Practical interpretation</h3>
              <ul>
                <li><strong>Category fraction + average:</strong> 0.73 means 73% of valid native pixels inside the target cell belonged to that class/group.</li>
                <li><strong>Original categorical + nearest:</strong> the target cell stores one class code. It is useful for dominant class maps, not for mixtures.</li>
                <li><strong>Original categorical + mode:</strong> the target cell stores the most frequent class. It still loses minority classes.</li>
                <li><strong>Class mask:</strong> 1 means the final aligned raster equals the chosen code. It is good for masks and distance-to-class workflows.</li>
              </ul>
            </div>
          </>
        )}

        {section === "examples" && (
          <>
            <h2>Useful examples</h2>
            <div className="tutorial-grid examples-grid">
              <article>
                <strong>Broadleaf forest cover at 100 m</strong>
                <span>Add official categorical land-cover layer, select the broadleaf category fraction, keep resampling as average, and optionally do not add the original categorical code.</span>
              </article>
              <article>
                <strong>Relative altitude</strong>
                <span>Select DEM, build a focal mean with a radius matching the neighbourhood, then use advanced expression x - y.</span>
              </article>
              <article>
                <strong>Distance to tracks</strong>
                <span>Use OSM tracks/forest roads as a binary/vector raster source, then spatial operation distance to mask.</span>
              </article>
              <article>
                <strong>Thermal range</strong>
                <span>Use guided recipe with tmax and tmin layers selected for matching temporal periods or aggregations.</span>
              </article>
              <article>
                <strong>Snow persistence</strong>
                <span>Use snow-days and valid-days layers or aggregations. The recipe divides snow observations by valid observations.</span>
              </article>
            </div>
            <div className="tutorial-subsection">
              <h3>Bear dataset examples</h3>
              <ul>
                <li><strong>Slope:</strong> build a single feature with spatial operation DEM terrain and method slope from a DEM input.</li>
                <li><strong>Ruggedness:</strong> build a DEM terrain feature with ruggedness or roughness and choose a radius in cells that matches the ecological scale.</li>
                <li><strong>Tree cover density:</strong> use an official continuous CLMS forest/tree-cover variable and choose average resampling to 100 m.</li>
                <li><strong>Shrubland/grassland/rock fractions:</strong> use category fractions from the best available land-cover layer, not class masks.</li>
                <li><strong>Human pressure:</strong> use OSM roads/tracks/settlements as masks or vector-derived rasters, then create distance features.</li>
              </ul>
            </div>
          </>
        )}

        {section === "limits" && (
          <>
            <h2>Limitations and sharp edges</h2>
            <ul>
              <li>Unavailable source combinations can still fail during download if the provider does not publish that exact file.</li>
              <li>Categorical temporal aggregation is only meaningful for supplied layers, mode-like summaries, or category fractions; numeric mean over class codes is not interpretable.</li>
              <li>Class masks work after source alignment. They are binary final-grid masks, not coverage percentages.</li>
              <li>Very large selections can generate many rasters. Check the output count and validation warnings before running.</li>
              <li>The workbench builds raster datasets. It does not train habitat models or XGBoost models itself.</li>
            </ul>
            <div className="tutorial-subsection">
              <h3>What to check before launching a long run</h3>
              <ul>
                <li>Every final feature name should be unique after temporal and dimension expansion.</li>
                <li>Every temporal aggregation should use years/months actually available in the source.</li>
                <li>Every categorical percentage should be a category fraction with average resampling.</li>
                <li>Every source with multiple native resolutions should use the intended original resolution.</li>
                <li>The Review page should show only variables you intend to keep in the final dataset.</li>
              </ul>
            </div>
          </>
        )}
      </section>
    </main>
  );
}

function MapBboxPicker({
  bounds,
  footprint,
  displayCrs,
  onChange
}: {
  bounds: AoiBounds | null;
  footprint?: LonLatPoint[] | null;
  displayCrs?: string;
  onChange: (bounds: AoiBounds) => void;
}) {
  const [mapStyle, setMapStyle] = useState<"street" | "satellite">("street");
  const [tool, setTool] = useState<"pan" | "draw">("pan");
  const [zoom, setZoom] = useState(aoiInitialMapZoom);
  const [center, setCenter] = useState({ lon: 0.55, lat: 42.45 });
  const [panStart, setPanStart] = useState<{
    clientX: number;
    clientY: number;
    centerPixel: { x: number; y: number };
  } | null>(null);
  const [showLayerMenu, setShowLayerMenu] = useState(false);
  const [dragStart, setDragStart] = useState<{ lon: number; lat: number } | null>(null);
  const [dragEnd, setDragEnd] = useState<{ lon: number; lat: number } | null>(null);

  useEffect(() => {
    if (!bounds) return;
    const safeBounds = orderedBounds(bounds);
    const nextCenter = {
      lon: clamp((safeBounds.xmin + safeBounds.xmax) / 2, pyreneesWgs84Envelope.xmin, pyreneesWgs84Envelope.xmax),
      lat: clamp((safeBounds.ymin + safeBounds.ymax) / 2, pyreneesWgs84Envelope.ymin, pyreneesWgs84Envelope.ymax)
    };
    setCenter((current) =>
      Math.abs(current.lon - nextCenter.lon) < 1e-8 && Math.abs(current.lat - nextCenter.lat) < 1e-8
        ? current
        : nextCenter
    );
  }, [bounds?.xmin, bounds?.xmax, bounds?.ymin, bounds?.ymax]);

  const mapPixels = useMemo(() => {
    const centerPixel = lonLatToPixel(center.lon, center.lat, zoom);
    return {
      left: centerPixel.x - aoiMapViewport.width / 2,
      top: centerPixel.y - aoiMapViewport.height / 2,
      width: aoiMapViewport.width,
      height: aoiMapViewport.height
    };
  }, [center, zoom]);
  const tiles = useMemo(() => {
    const minX = Math.floor(mapPixels.left / 256);
    const maxX = Math.floor((mapPixels.left + mapPixels.width) / 256);
    const minY = Math.floor(mapPixels.top / 256);
    const maxY = Math.floor((mapPixels.top + mapPixels.height) / 256);
    const rows: Array<{ x: number; y: number; left: number; top: number; width: number; height: number }> = [];
    for (let x = minX; x <= maxX; x += 1) {
      for (let y = minY; y <= maxY; y += 1) {
        rows.push({
          x,
          y,
          left: ((x * 256 - mapPixels.left) / mapPixels.width) * 100,
          top: ((y * 256 - mapPixels.top) / mapPixels.height) * 100,
          width: (256 / mapPixels.width) * 100,
          height: (256 / mapPixels.height) * 100
        });
      }
    }
    return rows;
  }, [mapPixels]);

  function pointFromEvent(event: ReactPointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const localX = clamp(event.clientX - rect.left, 0, rect.width);
    const localY = clamp(event.clientY - rect.top, 0, rect.height);
    const pixelX = mapPixels.left + (localX / rect.width) * mapPixels.width;
    const pixelY = mapPixels.top + (localY / rect.height) * mapPixels.height;
    const lonLat = pixelToLonLat(pixelX, pixelY, zoom);
    return {
      lon: clamp(lonLat.lon, pyreneesWgs84Envelope.xmin, pyreneesWgs84Envelope.xmax),
      lat: clamp(lonLat.lat, pyreneesWgs84Envelope.ymin, pyreneesWgs84Envelope.ymax)
    };
  }

  function rectangleFootprint(nextBounds: AoiBounds): LonLatPoint[] {
    const safeBounds = orderedBounds(nextBounds);
    return [
      { lon: safeBounds.xmin, lat: safeBounds.ymin },
      { lon: safeBounds.xmax, lat: safeBounds.ymin },
      { lon: safeBounds.xmax, lat: safeBounds.ymax },
      { lon: safeBounds.xmin, lat: safeBounds.ymax }
    ];
  }

  function overlayPath(points: LonLatPoint[]) {
    if (points.length < 3) return "";
    const commands = points.map((point, index) => {
      const pixel = lonLatToPixel(point.lon, point.lat, zoom);
      const x = pixel.x - mapPixels.left;
      const y = pixel.y - mapPixels.top;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    });
    return `${commands.join(" ")} Z`;
  }

  const draftBounds = dragStart && dragEnd
    ? orderedBounds({
        xmin: dragStart.lon,
        xmax: dragEnd.lon,
        ymin: dragStart.lat,
        ymax: dragEnd.lat
      })
    : null;
  const shownBounds = draftBounds ?? bounds;
  const shownFootprint = draftBounds
    ? rectangleFootprint(draftBounds)
    : (footprint && footprint.length > 0 ? footprint : shownBounds ? rectangleFootprint(shownBounds) : null);
  const tileBase = mapStyle === "street"
    ? "https://tile.openstreetmap.org"
    : "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile";
  const path = shownFootprint ? overlayPath(shownFootprint) : "";

  return (
    <div className="bbox-map-panel">
      <div
        className={`bbox-map ${mapStyle} ${tool === "draw" ? "drawing" : "panning"}`}
        onPointerDown={(event) => {
          if (tool === "draw") {
            const point = pointFromEvent(event);
            setDragStart(point);
            setDragEnd(point);
          } else {
            setPanStart({
              clientX: event.clientX,
              clientY: event.clientY,
              centerPixel: lonLatToPixel(center.lon, center.lat, zoom)
            });
          }
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (tool === "draw") {
            if (!dragStart) return;
            setDragEnd(pointFromEvent(event));
            return;
          }
          if (!panStart) return;
          const rect = event.currentTarget.getBoundingClientRect();
          const scaleX = mapPixels.width / rect.width;
          const scaleY = mapPixels.height / rect.height;
          const nextPixel = {
            x: panStart.centerPixel.x - (event.clientX - panStart.clientX) * scaleX,
            y: panStart.centerPixel.y - (event.clientY - panStart.clientY) * scaleY
          };
          const nextCenter = pixelToLonLat(nextPixel.x, nextPixel.y, zoom);
          setCenter({
            lon: clamp(nextCenter.lon, pyreneesWgs84Envelope.xmin, pyreneesWgs84Envelope.xmax),
            lat: clamp(nextCenter.lat, pyreneesWgs84Envelope.ymin, pyreneesWgs84Envelope.ymax)
          });
        }}
        onPointerUp={(event) => {
          if (tool === "draw" && dragStart) {
            const point = pointFromEvent(event);
            const next = orderedBounds({
              xmin: dragStart.lon,
              xmax: point.lon,
              ymin: dragStart.lat,
              ymax: point.lat
            });
            if (Math.abs(next.xmax - next.xmin) > 0.001 && Math.abs(next.ymax - next.ymin) > 0.001) {
              onChange({
                xmin: formatCoord(next.xmin),
                xmax: formatCoord(next.xmax),
                ymin: formatCoord(next.ymin),
                ymax: formatCoord(next.ymax)
              });
            }
          }
          setDragStart(null);
          setDragEnd(null);
          setPanStart(null);
        }}
      >
        <div className="map-control-stack zoom-controls" onPointerDown={(event) => event.stopPropagation()}>
          <button title="Zoom in" onClick={(event) => { event.stopPropagation(); setZoom((current) => Math.min(12, current + 1)); }}>+</button>
          <button title="Zoom out" onClick={(event) => { event.stopPropagation(); setZoom((current) => Math.max(6, current - 1)); }}>-</button>
        </div>
        <div className="map-control-stack map-tool-controls" onPointerDown={(event) => event.stopPropagation()}>
          <button
            title="Move map"
            className={tool === "pan" ? "active" : ""}
            onClick={(event) => { event.stopPropagation(); setTool("pan"); }}
          >
            ↕
          </button>
          <button
            title="Draw bounding box"
            className={tool === "draw" ? "active" : ""}
            onClick={(event) => { event.stopPropagation(); setTool("draw"); }}
          >
            ▭
          </button>
          <button
            title="Map layers"
            className={showLayerMenu ? "active" : ""}
            onClick={(event) => { event.stopPropagation(); setShowLayerMenu((current) => !current); }}
          >
            ▧
          </button>
        </div>
        {showLayerMenu && (
          <div className="map-layer-menu" onPointerDown={(event) => event.stopPropagation()}>
            <button className={mapStyle === "street" ? "active" : ""} onClick={() => { setMapStyle("street"); setShowLayerMenu(false); }}>Map</button>
            <button className={mapStyle === "satellite" ? "active" : ""} onClick={() => { setMapStyle("satellite"); setShowLayerMenu(false); }}>Satellite</button>
          </div>
        )}
        {tiles.map((tile) => (
          <img
            key={`${mapStyle}-${tile.x}-${tile.y}`}
            alt=""
            draggable={false}
            src={mapStyle === "street"
              ? `${tileBase}/${zoom}/${tile.x}/${tile.y}.png`
              : `${tileBase}/${zoom}/${tile.y}/${tile.x}`}
            style={{
              left: `${tile.left}%`,
              top: `${tile.top}%`,
              width: `${tile.width}%`,
              height: `${tile.height}%`
            }}
          />
        ))}
        {path && (
          <svg
            className="bbox-overlay"
            viewBox={`0 0 ${mapPixels.width} ${mapPixels.height}`}
            preserveAspectRatio="none"
            aria-label={`Projected AOI footprint${displayCrs ? ` ${displayCrs}` : ""}`}
          >
            <path className="bbox-selection" d={path} />
          </svg>
        )}
        <div className="map-attribution">
          {mapStyle === "street" ? "© OpenStreetMap contributors" : "Esri World Imagery"}
        </div>
      </div>
      {bounds && (
        <div className="bbox-readout">
          <span>xmin {bounds.xmin}</span>
          <span>xmax {bounds.xmax}</span>
          <span>ymin {bounds.ymin}</span>
          <span>ymax {bounds.ymax}</span>
        </div>
      )}
    </div>
  );
}

function AoiBuilderPanel({
  projectConfig,
  resolutions,
  targetCrs,
  setTargetCrs,
  aoiPath,
  setAoiPath,
  resolution,
  setResolution,
  onAoiCreated
}: {
  projectConfig: string;
  resolutions: number[];
  targetCrs: string;
  setTargetCrs: (value: string) => void;
  aoiPath: string;
  setAoiPath: (value: string) => void;
  resolution: number;
  setResolution: (value: number) => void;
  onAoiCreated: (aoi: AoiCatalog) => void;
}) {
  const [aoiForm, setAoiForm] = useState({
    name: "custom_aoi",
    description: "Workbench-created AOI.",
    crs: targetCrs,
    xmin: "",
    xmax: "",
    ymin: "",
    ymax: ""
  });
  const [aoiStatus, setAoiStatus] = useState<string | null>(null);
  const [gridStatus, setGridStatus] = useState<string | null>(null);
  const normalizedAoiCrs = normalizeCrsCode(aoiForm.crs);
  const bounds = {
    xmin: Number(aoiForm.xmin),
    xmax: Number(aoiForm.xmax),
    ymin: Number(aoiForm.ymin),
    ymax: Number(aoiForm.ymax)
  };
  const boundsAreNumeric = Object.values(bounds).every(Number.isFinite);
  const boundsAreOrdered = boundsAreNumeric && bounds.xmin < bounds.xmax && bounds.ymin < bounds.ymax;
  const mapProjectionSupported = canProjectToMap(aoiForm.crs);
  const mapBounds = boundsAreOrdered && mapProjectionSupported ? boundsToWgs84(bounds, aoiForm.crs) : null;
  const mapFootprint = boundsAreOrdered && mapProjectionSupported ? boundsFootprintToWgs84(bounds, aoiForm.crs) : null;
  const insidePyreneesBuffer = !boundsAreOrdered || !mapProjectionSupported || !mapBounds || (
    mapBounds.xmin >= pyreneesWgs84Envelope.xmin &&
    mapBounds.xmax <= pyreneesWgs84Envelope.xmax &&
    mapBounds.ymin >= pyreneesWgs84Envelope.ymin &&
    mapBounds.ymax <= pyreneesWgs84Envelope.ymax
  );
  const resolutionChecks = boundsAreOrdered
    ? resolutionChecksForBounds(bounds, aoiForm.crs, resolutions)
    : [];
  const allResolutionChecksPass = resolutionChecks.length > 0 &&
    resolutionChecks.every((check) => check.widthOk && check.heightOk);
  const canCreateAoi = aoiForm.name.trim().length > 0 && aoiForm.crs.trim().length > 0 && boundsAreOrdered && insidePyreneesBuffer;

  function applyResolutionRebounding() {
    if (!boundsAreOrdered) return;
    const next = expandBoundsToResolutions(bounds, aoiForm.crs, resolutions);
    setAoiForm({
      ...aoiForm,
      xmin: String(next.xmin),
      xmax: String(next.xmax),
      ymin: String(next.ymin),
      ymax: String(next.ymax)
    });
  }

  async function submitAoi() {
    if (!canCreateAoi) return;
    setAoiStatus(null);
    try {
      const result = await createAoiConfig({
        name: aoiForm.name,
        description: aoiForm.description,
        crs: normalizedAoiCrs,
        bounds
      });
      onAoiCreated(result.aoi);
      setTargetCrs(result.aoi.crs ?? targetCrs);
      setAoiPath(result.aoi.path);
      setAoiStatus(`Created ${result.aoi.path}`);
    } catch (error) {
      setAoiStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function submitGrid() {
    setGridStatus(null);
    try {
      const result = await createProjectGrid({
        project_config: projectConfig,
        aoi_config: aoiPath,
        crs: targetCrs,
        resolution_m: resolution,
        overwrite: false
      });
      setGridStatus(`Grid ready at ${result.grid_path}`);
    } catch (error) {
      setGridStatus(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <main className="workspace two-col aoi-workspace">
      <section className="panel">
        <h2>Create AOI Config</h2>
        <div className="form-grid">
          <label>
            AOI name
            <input value={aoiForm.name} onChange={(event) => setAoiForm({ ...aoiForm, name: event.target.value })} />
          </label>
          <label>
            AOI CRS
            <select value={normalizedAoiCrs} onChange={(event) => setAoiForm({ ...aoiForm, crs: event.target.value })}>
              <option value="EPSG:3035">EPSG:3035</option>
              <option value="EPSG:4326">EPSG:4326</option>
            </select>
          </label>
          <label>
            Target grid resolution
            <select value={resolution} onChange={(event) => setResolution(Number(event.target.value))}>
              {!resolutions.includes(resolution) && <option value={resolution}>{resolution} m</option>}
              {resolutions.map((item) => (
                <option key={item} value={item}>{item} m</option>
              ))}
            </select>
          </label>
          <label className="span-2">
            Description
            <input value={aoiForm.description} onChange={(event) => setAoiForm({ ...aoiForm, description: event.target.value })} />
          </label>
          {(["xmin", "xmax", "ymin", "ymax"] as const).map((key) => (
            <label key={key}>
              {key}
              <input type="number" value={aoiForm[key]} onChange={(event) => setAoiForm({ ...aoiForm, [key]: event.target.value })} />
            </label>
          ))}
        </div>
        {!boundsAreOrdered && <div className="notice info">Bounds must satisfy xmin &lt; xmax and ymin &lt; ymax.</div>}
        {boundsAreOrdered && !insidePyreneesBuffer && (
          <div className="notice error">Bounds must stay inside the broad Pyrenees working envelope.</div>
        )}
        {boundsAreOrdered && mapProjectionSupported && insidePyreneesBuffer && (
          <div className="notice success">Bounds are inside the broad Pyrenees working envelope.</div>
        )}
        {boundsAreOrdered && (
          <div className={`resolution-check-panel ${allResolutionChecksPass ? "ok" : "warn"}`}>
            <div className="resolution-check-head">
              <strong>Resolution divisibility</strong>
              <small>{normalizedAoiCrs === "EPSG:4326" ? "Estimated from WGS84 bounds" : `Checked in ${normalizedAoiCrs}`}</small>
            </div>
            <div className="resolution-check-list">
              {resolutionChecks.map((check) => (
                <span key={check.resolution} className={check.widthOk && check.heightOk ? "ok" : "warn"}>
                  {check.resolution} m {check.widthOk && check.heightOk ? "accepted" : "needs rebounding"}
                </span>
              ))}
            </div>
          </div>
        )}
        <div className="button-row aoi-actions">
          <button className="ghost" disabled={!boundsAreOrdered || allResolutionChecksPass} onClick={applyResolutionRebounding}>
            Apply resolution rebounding
          </button>
          <button className="primary" disabled={!canCreateAoi} onClick={submitAoi}>Create AOI config</button>
          <button className="ghost" onClick={submitGrid}>Create target grid</button>
        </div>
        {aoiStatus && <div className="notice info">{aoiStatus}</div>}
        {gridStatus && <div className="notice info">{gridStatus}</div>}
      </section>
      <section className="panel">
        <h2>Map Preview</h2>
        <MapBboxPicker
          bounds={mapBounds}
          footprint={mapFootprint}
          displayCrs={normalizedAoiCrs}
          onChange={(nextBounds) => {
            const targetBounds = boundsFromWgs84(nextBounds, normalizedAoiCrs);
            if (!targetBounds) return;
            setAoiForm({
              ...aoiForm,
              crs: normalizedAoiCrs,
              xmin: String(formatCrsCoord(targetBounds.xmin, normalizedAoiCrs)),
              xmax: String(formatCrsCoord(targetBounds.xmax, normalizedAoiCrs)),
              ymin: String(formatCrsCoord(targetBounds.ymin, normalizedAoiCrs)),
              ymax: String(formatCrsCoord(targetBounds.ymax, normalizedAoiCrs))
            });
          }}
        />
        {!mapProjectionSupported && (
          <div className="notice info compact-notice">
            Map drawing currently supports EPSG:4326 and EPSG:3035. Other CRS values can still be typed manually.
          </div>
        )}
      </section>
    </main>
  );
}

function FeatureProjectSetupPanel({
  catalog,
  aois,
  runName,
  setRunName,
  description,
  setDescription,
  projectConfig,
  setProjectConfig,
  targetCrs,
  setTargetCrs,
  aoiPath,
  setAoiPath,
  resolution,
  setResolution,
  stages,
  setStages,
  datasetDir,
  setDatasetDir,
  onNext
}: ProjectPanelProps & { onNext: () => void }) {
  const resolutions = catalog?.project.available_resolutions_m ?? [100];
  const supportedStages = catalog?.supported_stages ?? ["download", "clip", "build", "all"];

  return (
    <main className="workspace feature-project-setup">
      <section className="panel feature-setup-hero">
        <div>
          <span className="eyebrow">Project setup</span>
          <h2>Dataset workspace</h2>
          <p>Define the run envelope once, then build the final features one by one.</p>
        </div>
      </section>

      <section className="panel project-setup-card">
        <div className="panel-head">
          <h2>Run identity</h2>
          <span className="field-hint">Names and output folder</span>
        </div>
        <div className="form-grid project-setup-grid">
          <label>
            Run name
            <input value={runName} onChange={(event) => setRunName(event.target.value)} />
          </label>
          <label>
            Dataset directory
            <input value={datasetDir} onChange={(event) => setDatasetDir(event.target.value)} />
          </label>
          <label className="span-2">
            Description
            <textarea value={description} rows={3} onChange={(event) => setDescription(event.target.value)} />
          </label>
        </div>
      </section>

      <section className="panel project-setup-card">
        <div className="panel-head">
          <h2>Grid and AOI</h2>
          <span className="field-hint">Target geometry for all final features</span>
        </div>
        <div className="form-grid project-setup-grid">
          <label>
            Target CRS
            <select value={targetCrs} onChange={(event) => setTargetCrs(event.target.value)}>
              <option value="EPSG:3035">EPSG:3035</option>
              <option value="EPSG:4326">EPSG:4326</option>
            </select>
          </label>
          <label>
            AOI
            <select value={aoiPath} onChange={(event) => setAoiPath(event.target.value)}>
              {aois.map((aoi) => (
                <option key={aoi.path} value={aoi.path}>{aoi.name}</option>
              ))}
            </select>
          </label>
          <label>
            Target resolution
            <select value={resolution} onChange={(event) => setResolution(Number(event.target.value))}>
              {!resolutions.includes(resolution) && <option value={resolution}>{resolution} m</option>}
              {resolutions.map((item) => (
                <option key={item} value={item}>{item} m</option>
              ))}
            </select>
          </label>
          <label>
            Project config
            <input value={projectConfig} onChange={(event) => setProjectConfig(event.target.value)} />
          </label>
        </div>
      </section>

      <section className="panel project-setup-card">
        <div className="panel-head">
          <h2>Stages</h2>
          <span className="field-hint">Run all by default</span>
        </div>
        <div className="choice-list compact">
          {supportedStages.map((stage) => (
            <label key={stage} className="check-row">
              <input
                type="checkbox"
                checked={stages.includes(stage)}
                onChange={() => setStages(toggleStage(stages, stage))}
              />
              <span>{stage}</span>
            </label>
          ))}
        </div>
      </section>

      <section className="panel project-start-panel">
        <div>
          <h2>Ready for features</h2>
          <p className="builder-copy">When these project-level settings look right, continue to build the final dataset variables.</p>
        </div>
        <button className="primary" onClick={onNext}>Start creating features</button>
      </section>
    </main>
  );
}

function featureOutputNames(feature: DatasetFeatureConfig) {
  return (feature.outputs && feature.outputs.length > 0)
    ? feature.outputs.map((output) => output.name)
    : [feature.name];
}

function featureOutputCount(feature: DatasetFeatureConfig) {
  return featureOutputNames(feature).length;
}

function FeatureSidebar({
  features,
  setFeatures,
  onEdit
}: {
  features: DatasetFeatureConfig[];
  setFeatures: (features: DatasetFeatureConfig[]) => void;
  onEdit: (feature: DatasetFeatureConfig, index: number) => void;
}) {
  return (
    <aside className="feature-sidebar panel">
      <div className="panel-head">
        <h2>Final features</h2>
        <span className="field-hint">{features.length} cards</span>
      </div>
      <div className="final-feature-list">
        {features.map((feature, index) => (
          <article className="final-feature-card" key={`${feature.name}-${index}`}>
            <div>
              <strong>{feature.title || feature.name}</strong>
              <small>{feature.name} · {feature.build_type} · {featureOutputCount(feature)} output{featureOutputCount(feature) === 1 ? "" : "s"}</small>
            </div>
            {feature.outputs && feature.outputs.length > 1 && (
              <div className="feature-chip-row">
                {feature.outputs.slice(0, 8).map((output) => (
                  <span key={output.name}>{output.suffix || output.name}</span>
                ))}
                {feature.outputs.length > 8 && <span>+{feature.outputs.length - 8}</span>}
              </div>
            )}
            <div className="button-row">
              <button className="ghost" onClick={() => onEdit(feature, index)}>Edit</button>
              <button
                className="ghost danger"
                onClick={() => {
                  const outputs = new Set(featureOutputNames(feature));
                  const dependents = features
                    .filter((candidate, candidateIndex) => candidateIndex !== index)
                    .filter((candidate) =>
                      Object.values(candidate.inputs ?? {}).some((input) =>
                        input.kind === "feature" && outputs.has(input.output || input.feature)
                      )
                    )
                    .map((candidate) => candidate.name);
                  if (dependents.length > 0) {
                    const ok = window.confirm(
                      `Removing ${feature.name} will also remove dependent features: ${dependents.join(", ")}. Continue?`
                    );
                    if (!ok) return;
                    setFeatures(features.filter((candidate, candidateIndex) =>
                      candidateIndex !== index && !dependents.includes(candidate.name)
                    ));
                    return;
                  }
                  setFeatures(features.filter((_, candidateIndex) => candidateIndex !== index));
                }}
              >
                Remove
              </button>
            </div>
          </article>
        ))}
        {features.length === 0 && (
          <div className="empty-state">No final features yet.</div>
        )}
      </div>
    </aside>
  );
}

function FeatureInputPicker({
  catalog,
  existingFeatures,
  projectName,
  allowExistingFeatures = true,
  allowCategoryFractions = true,
  filter,
  onCancel,
  onConfirm
}: {
  catalog: WorkbenchCatalog;
  existingFeatures: DatasetFeatureConfig[];
  projectName: string;
  allowExistingFeatures?: boolean;
  allowCategoryFractions?: boolean;
  filter?: (variable: VariableCatalog, source: SourceCatalog) => boolean;
  onCancel: () => void;
  onConfirm: (bundle: InputBundle) => void;
}) {
  const providerGroups = useMemo(() => sourceProviderGroups(catalog), [catalog]);
  const [step, setStep] = useState<FeaturePickerStep>("origin");
  const [origin, setOrigin] = useState<"project" | "official" | null>(null);
  const [providerId, setProviderId] = useState("");
  const providerGroup = providerGroups.find((group) => group.provider === providerId);
  const [sourceId, setSourceId] = useState("");
  const source = catalog.sources.find((item) => item.id === sourceId);
  const variables = source ? sourceVariables(source).filter((variable) => !filter || filter(variable, source)) : [];
  const [variableName, setVariableName] = useState("");
  const variable = variables.find((item) => item.name === variableName);
  const [dimensions, setDimensions] = useState<Record<string, string[]>>({});
  const [temporalMode, setTemporalMode] = useState<FeatureTemporalMode>("static");
  const [temporalLayers, setTemporalLayers] = useState<TemporalSelection["layers"]>({
    annual: false,
    annual_index: false,
    months: [],
    seasons: [],
    years: []
  });
  const [aggregation, setAggregation] = useState<CustomAggregation>(aggregationDefaults(undefined, []));
  const [aggregations, setAggregations] = useState<CustomAggregation[]>([]);
  const [selectedCategoryTokens, setSelectedCategoryTokens] = useState<string[]>([]);
  const [resamplingMethod, setResamplingMethod] = useState("nearest");
  const [sourceResolution, setSourceResolution] = useState("");

  useEffect(() => {
    if (!source) return;
    const nextDimensions: Record<string, string[]> = {};
    for (const [key] of sourceDimensionEntries(source)) {
      nextDimensions[key] = [];
    }
    setDimensions(nextDimensions);
    setVariableName("");
    setSelectedCategoryTokens([]);
    setTemporalMode(defaultTemporalModeForSource(source));
    setTemporalLayers(initialTemporalLayers(source));
    setAggregation(aggregationDefaults(source, []));
    setAggregations([]);
    setResamplingMethod("nearest");
    setSourceResolution(source.source_resolution ?? sourceResolutionChoices(source)[0] ?? "");
  }, [source?.id]);

  useEffect(() => {
    if (!variable || !source) return;
    setAggregation((current) => ({
      ...aggregationDefaults(source, [variable.name], current.metric),
      name: current.name || "period_mean"
    }));
    setAggregations([]);
    setResamplingMethod(defaultResamplingForOutput(variable, selectedCategoryTokens.length > 0));
  }, [source?.id, variable?.name, selectedCategoryTokens.length]);

  const sourceDimensionsComplete = dimensionsAreComplete(source, dimensions);
  const temporalComplete = temporalSelectionIsComplete(source, temporalMode, temporalLayers, aggregation.name, aggregation, aggregations, dimensions);
  const aggregationYearBounds = effectiveAggregationYearBounds(source, dimensions);
  const canAddAggregation = aggregationDraftIsComplete(source, dimensions, aggregation)
    && !aggregations.some((item) => item.name === aggregation.name);
  const categoryClasses = allowCategoryFractions ? variable?.category_classes ?? [] : [];
  const selectedCategoryFractions = categoryClasses
    .filter((item) => selectedCategoryTokens.includes(categoryClassToken(item)))
    .map((item) => ({
      ...categoryFractionConfig(variable as VariableCatalog, item),
      resampling: resamplingMethod
    }));
  const canConfirmOfficial = Boolean(source && variable && sourceDimensionsComplete && temporalComplete);
  const waitingForAddedAggregation = step === "temporal"
    && (temporalMode === "aggregate" || temporalMode === "postprocess_aggregate")
    && aggregations.length === 0;

  function chooseProjectOutput(feature: DatasetFeatureConfig, outputName: string) {
    onConfirm({
      label: `${projectName} · ${outputName}`,
      outputs: [{
        name: outputName,
        label: `${feature.title || feature.name} · ${outputName}`,
        input: { kind: "feature", feature: outputName, output: outputName },
        suffix: outputName
      }]
    });
  }

  function officialBundle(): InputBundle | null {
    if (!source || !variable || !canConfirmOfficial) return null;
    const categories = selectedCategoryFractions.length > 0 ? selectedCategoryFractions : [undefined];
    const activeAggregations = (temporalMode === "aggregate" || temporalMode === "postprocess_aggregate")
      ? aggregations
      : [aggregation];
    const outputs = categories.flatMap((categoryFraction) =>
      activeAggregations.flatMap((item) => buildSourceLayerOutputs({
        featureName: categoryFraction?.name ?? variable.name,
        source,
        variable,
        dimensions,
        temporalMode,
        years: temporalLayers.years,
        layers: temporalLayers,
        aggregation: item,
        categoryFraction,
        sourceResolution: sourceResolution || source.source_resolution,
        resampling: resamplingMethod
      }))
    );
    const options = outputs
      .filter((output) => output.source)
      .map((output) => ({
        name: output.name,
        label: `${sourceShortName(source)} · ${output.name}`,
        input: output.source as DatasetFeatureInput,
        suffix: output.suffix,
        temporalKey: output.temporal_key,
        dimensionKey: output.dimension_key,
        variable
      }));
    if (options.length === 0) return null;
    return {
      label: `${sourceShortName(source)} · ${variable.name}${options.length > 1 ? ` · ${options.length} outputs` : ""}`,
      outputs: options
    };
  }

  function goBack() {
    if (step === "origin") return;
    if (step === "provider") setStep("origin");
    if (step === "source") setStep("provider");
    if (step === "variable") setStep("source");
    if (step === "category") setStep("variable");
    if (step === "dimensions") setStep(categoryClasses.length > 0 ? "category" : "variable");
    if (step === "temporal") setStep(sourceDimensionEntries(source as SourceCatalog).length > 0 ? "dimensions" : categoryClasses.length > 0 ? "category" : "variable");
    if (step === "resampling") setStep("temporal");
  }

  function stepAfterVariable(selectedSource: SourceCatalog, selectedVariable: VariableCatalog) {
    const hasCategories = allowCategoryFractions && (selectedVariable.category_classes ?? []).length > 0;
    if (hasCategories) return "category";
    if (sourceDimensionEntries(selectedSource).length > 0) return "dimensions";
    return "temporal";
  }

  function goNext() {
    if (step === "origin" && origin === "official") setStep("provider");
    if (step === "provider" && providerId) setStep("source");
    if (step === "source" && sourceId) setStep("variable");
    if (step === "variable" && variable && source) setStep(stepAfterVariable(source, variable));
    if (step === "category") setStep(sourceDimensionEntries(source as SourceCatalog).length > 0 ? "dimensions" : "temporal");
    if (step === "dimensions" && sourceDimensionsComplete) setStep("temporal");
    if (step === "temporal" && temporalComplete) setStep("resampling");
  }

  function canGoNext() {
    if (step === "origin") return origin === "official";
    if (step === "provider") return Boolean(providerId);
    if (step === "source") return Boolean(sourceId);
    if (step === "variable") return Boolean(variable);
    if (step === "category") return true;
    if (step === "dimensions") return sourceDimensionsComplete;
    if (step === "temporal") return temporalComplete;
    return false;
  }

  function confirmOfficial() {
    const bundle = officialBundle();
    if (bundle) onConfirm(bundle);
  }

  function addAggregation() {
    if (!canAddAggregation) return;
    setAggregations([...aggregations, { ...aggregation }]);
    setAggregation({
      ...aggregation,
      name: `${aggregation.name}_${aggregations.length + 2}`
    });
  }

  return (
    <div className="modal-backdrop">
      <section className="feature-picker panel feature-picker-sheet">
        <div className="panel-head">
          <h2>Select input layer</h2>
          <div className="button-row">
            {step !== "origin" && <button className="ghost" onClick={goBack}>Back</button>}
            <button className="ghost" onClick={onCancel}>Cancel</button>
          </div>
        </div>

        <div className="feature-picker-steps">
          {["origin", "provider", "source", "variable", "category", "dimensions", "temporal", "resampling"].map((item) => (
            <span key={item} className={step === item ? "active" : ""}>{humanizeId(item)}</span>
          ))}
        </div>

        {step === "origin" && (
          <div className="feature-tool-grid picker-step-panel">
            {allowExistingFeatures && existingFeatures.length > 0 && (
              <button
                className={`feature-tool-card ${origin === "project" ? "active" : ""}`}
                onClick={() => setOrigin("project")}
                onDoubleClick={() => setOrigin("project")}
              >
                <strong>{projectName}</strong>
                <small>Use a final feature already created in this project.</small>
              </button>
            )}
            <button
              className={`feature-tool-card ${origin === "official" ? "active" : ""}`}
              onClick={() => setOrigin("official")}
              onDoubleClick={() => {
                setOrigin("official");
                setStep("provider");
              }}
            >
              <strong>Official sources</strong>
              <small>Select a source, product, variable, dimensions and temporal processing.</small>
            </button>
          </div>
        )}

        {step === "origin" && origin === "project" && (
          <div className="picker-block">
            <h3>{projectName}</h3>
            <div className="choice-list compact project-input-list">
              {existingFeatures.flatMap((feature) =>
                featureOutputNames(feature).map((output) => (
                  <button key={`${feature.name}-${output}`} className="source-select-button" onClick={() => chooseProjectOutput(feature, output)}>
                    <strong>{feature.title || feature.name}</strong>
                    <small>{output}</small>
                  </button>
                ))
              )}
            </div>
          </div>
        )}

        {step === "provider" && (
          <div className="picker-card-grid picker-step-panel">
            {providerGroups.map((group) => (
              <button
                key={group.provider}
                className={`source-select-button ${group.provider === providerId ? "active" : ""}`}
                onClick={() => {
                  setProviderId(group.provider);
                  setSourceId("");
                }}
                onDoubleClick={() => {
                  setProviderId(group.provider);
                  setSourceId("");
                  setStep("source");
                }}
              >
                <strong>{group.meta?.title ?? humanizeId(group.provider)}</strong>
                <small>{group.sources.length} products</small>
              </button>
            ))}
          </div>
        )}

        {step === "source" && (
          <div className="picker-card-grid picker-step-panel">
            {(providerGroup?.sources ?? []).map((item) => (
              <button
                key={item.id}
                className={`source-select-button ${item.id === sourceId ? "active" : ""}`}
                onClick={() => setSourceId(item.id)}
                onDoubleClick={() => {
                  setSourceId(item.id);
                  setStep("variable");
                }}
              >
                <strong>{sourceDisplayName(item)}</strong>
                <small>{item.id}</small>
              </button>
            ))}
          </div>
        )}

        {step === "variable" && (
          <div className="picker-card-grid picker-step-panel">
            {variables.map((item) => (
              <button
                key={item.name}
                className={`source-select-button ${item.name === variableName ? "active" : ""}`}
                onClick={() => setVariableName(item.name)}
                onDoubleClick={() => {
                  setVariableName(item.name);
                  if (source) setStep(stepAfterVariable(source, item));
                }}
              >
                <strong>{item.description || humanizeId(item.name)}</strong>
                <small>{item.name}{item.unit ? ` · ${item.unit}` : ""}{item.value_semantics ? ` · ${item.value_semantics}` : ""}</small>
              </button>
            ))}
            {variables.length === 0 && <div className="empty-state">No valid variables for this input.</div>}
          </div>
        )}

        {step === "category" && variable && (
          <div className="picker-step-panel">
            <div className="notice info compact-notice">
              Keep no class selected to use the original categorical layer. Select category fractions when you need
              coverage ratios: the class is converted to 0/1 at native resolution and then resampled, usually with
              average, so a 100 m cell can store 0.8 forest and 0.2 shrubland instead of one dominant code.
            </div>
            <div className="picker-card-grid">
              {categoryClasses.map((item) => {
                const token = categoryClassToken(item);
                return (
                  <label key={token} className="check-row rich">
                    <input
                      type="checkbox"
                      checked={selectedCategoryTokens.includes(token)}
                      onChange={() => setSelectedCategoryTokens(toggleValue(selectedCategoryTokens, token))}
                    />
                    <span>
                      <strong>{item.label || item.name || token}</strong>
                      <small>{categoryClassValues(item).join(", ")}</small>
                    </span>
                  </label>
                );
              })}
            </div>
          </div>
        )}

        {step === "dimensions" && source && (
          <div className="picker-step-panel dimension-step-grid">
            {sourceDimensionEntries(source).map(([key, values]) => (
              <div className="dimension-box" key={key}>
                <strong>{humanizeId(key)}</strong>
                <label className="check-row">
                  <input
                    type="checkbox"
                    checked={allValuesSelected(dimensions[key] ?? [], values)}
                    onChange={() => setDimensions({
                      ...dimensions,
                      [key]: allValuesSelected(dimensions[key] ?? [], values) ? [] : values
                    })}
                  />
                  <span>All</span>
                </label>
                {values.map((value) => (
                  <label key={value} className="check-row">
                    <input
                      type="checkbox"
                      checked={(dimensions[key] ?? []).includes(value)}
                      onChange={() => setDimensions({
                        ...dimensions,
                        [key]: toggleValue(dimensions[key] ?? [], value)
                      })}
                    />
                    <span>{value}</span>
                  </label>
                ))}
              </div>
            ))}
          </div>
        )}

        {step === "temporal" && source && (
          <div className="picker-step-panel temporal-step-grid">
            {source.temporal ? (
              <>
                <div className="dimension-box">
                  <strong>Temporal mode</strong>
                  {supportedTemporalModes(source).map((mode) => (
                    <label key={mode} className="check-row">
                      <input
                        type="radio"
                        name="input-temporal-mode"
                        checked={temporalMode === mode}
                        onChange={() => setTemporalMode(mode)}
                      />
                      <span>{humanizeId(mode)}</span>
                    </label>
                  ))}
                </div>
                {temporalMode === "supplied_layers" && (
                  <div className="dimension-box">
                    <strong>Supplied layers</strong>
                    {source.temporal.temporal_layers?.annual && (
                      <label className="check-row">
                        <input type="checkbox" checked={temporalLayers.annual} onChange={() => setTemporalLayers({ ...temporalLayers, annual: !temporalLayers.annual })} />
                        <span>annual</span>
                      </label>
                    )}
                    {source.temporal.temporal_layers?.annual_index && (
                      <label className="check-row">
                        <input type="checkbox" checked={temporalLayers.annual_index} onChange={() => setTemporalLayers({ ...temporalLayers, annual_index: !temporalLayers.annual_index })} />
                        <span>annual index</span>
                      </label>
                    )}
                    {(source.temporal.temporal_layers?.years ?? []).length > 1 && (
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={allValuesSelected(temporalLayers.years, source.temporal.temporal_layers?.years ?? [])}
                          onChange={() => {
                            const years = source.temporal?.temporal_layers?.years ?? [];
                            setTemporalLayers({
                              ...temporalLayers,
                              years: allValuesSelected(temporalLayers.years, years) ? [] : [...years]
                            });
                          }}
                        />
                        <span>All years</span>
                      </label>
                    )}
                    {(source.temporal.temporal_layers?.years ?? []).map((year) => (
                      <label key={year} className="check-row">
                        <input type="checkbox" checked={temporalLayers.years.includes(year)} onChange={() => setTemporalLayers({ ...temporalLayers, years: toggleValue(temporalLayers.years, year) })} />
                        <span>{year}</span>
                      </label>
                    ))}
                    {(source.temporal.temporal_layers?.months ?? []).length > 1 && (
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={allValuesSelected(temporalLayers.months, source.temporal.temporal_layers?.months ?? [])}
                          onChange={() => {
                            const months = source.temporal?.temporal_layers?.months ?? [];
                            setTemporalLayers({
                              ...temporalLayers,
                              months: allValuesSelected(temporalLayers.months, months) ? [] : [...months]
                            });
                          }}
                        />
                        <span>All months</span>
                      </label>
                    )}
                    {(source.temporal.temporal_layers?.months ?? []).map((month) => (
                      <label key={month} className="check-row">
                        <input type="checkbox" checked={temporalLayers.months.includes(month)} onChange={() => setTemporalLayers({ ...temporalLayers, months: toggleValue(temporalLayers.months, month) })} />
                        <span>{month}</span>
                      </label>
                    ))}
                    {(source.temporal.temporal_layers?.seasons ?? []).length > 1 && (
                      <label className="check-row">
                        <input
                          type="checkbox"
                          checked={allValuesSelected(temporalLayers.seasons, source.temporal.temporal_layers?.seasons ?? [])}
                          onChange={() => {
                            const seasons = source.temporal?.temporal_layers?.seasons ?? [];
                            setTemporalLayers({
                              ...temporalLayers,
                              seasons: allValuesSelected(temporalLayers.seasons, seasons) ? [] : [...seasons]
                            });
                          }}
                        />
                        <span>All seasons</span>
                      </label>
                    )}
                    {(source.temporal.temporal_layers?.seasons ?? []).map((season) => (
                      <label key={season} className="check-row">
                        <input type="checkbox" checked={temporalLayers.seasons.includes(season)} onChange={() => setTemporalLayers({ ...temporalLayers, seasons: toggleValue(temporalLayers.seasons, season) })} />
                        <span>{season}</span>
                      </label>
                    ))}
                  </div>
                )}
                {temporalMode === "raw_slices" && (
                  <div className="dimension-box temporal-aggregate-box">
                    <strong>Raw slices</strong>
                    <div className="mini-form-grid">
                      <label>
                        Start month
                        <input type="number" min={1} max={12} value={aggregation.months?.[0] ?? 1} onChange={(event) => setAggregation({ ...aggregation, months: [clamp(Number(event.target.value), 1, 12), aggregation.months?.[1] ?? 12] })} />
                      </label>
                      <label>
                        End month
                        <input type="number" min={1} max={12} value={aggregation.months?.[1] ?? 12} onChange={(event) => setAggregation({ ...aggregation, months: [aggregation.months?.[0] ?? 1, clamp(Number(event.target.value), 1, 12)] })} />
                      </label>
                    </div>
                    {source.temporal.kind === "year_month_series" && (
                      <div className="mini-form-grid">
                        <label>
                          Start year
                          <input
                            type="number"
                            min={aggregationYearBounds?.[0]}
                            max={aggregationYearBounds?.[1]}
                            value={aggregation.years?.[0] ?? aggregationYearBounds?.[0] ?? source.temporal.default_years?.[0] ?? ""}
                            onChange={(event) => setAggregation({
                              ...aggregation,
                              years: clampYearRangeToBounds(
                                [Number(event.target.value), aggregation.years?.[1] ?? Number(event.target.value)],
                                aggregationYearBounds
                              )
                            })}
                          />
                        </label>
                        <label>
                          End year
                          <input
                            type="number"
                            min={aggregationYearBounds?.[0]}
                            max={aggregationYearBounds?.[1]}
                            value={aggregation.years?.[1] ?? aggregationYearBounds?.[1] ?? source.temporal.default_years?.[1] ?? ""}
                            onChange={(event) => setAggregation({
                              ...aggregation,
                              years: clampYearRangeToBounds(
                                [aggregation.years?.[0] ?? Number(event.target.value), Number(event.target.value)],
                                aggregationYearBounds
                              )
                            })}
                          />
                        </label>
                      </div>
                    )}
                  </div>
                )}
                {(temporalMode === "aggregate" || temporalMode === "postprocess_aggregate") && (
                  <div className="dimension-box temporal-aggregate-box">
                    <strong>Aggregation</strong>
                    {aggregationYearBounds && (
                      <div className="notice info compact-notice">
                        Available year range after selected dimensions: {aggregationYearBounds[0]}-{aggregationYearBounds[1]}.
                      </div>
                    )}
                    <label>
                      Name
                      <input value={aggregation.name} onChange={(event) => setAggregation({ ...aggregation, name: sanitizeToken(event.target.value) })} />
                    </label>
                    <label>
                      Metric
                      <select value={aggregation.metric} onChange={(event) => setAggregation({ ...aggregation, metric: event.target.value })}>
                        {catalog.supported_metrics.map((metric) => <option key={metric} value={metric}>{metric}</option>)}
                      </select>
                    </label>
                    {aggregation.years && (
                      <div className="mini-form-grid">
                        <label>
                          Start year
                          <input
                            type="number"
                            min={aggregationYearBounds?.[0]}
                            max={aggregationYearBounds?.[1]}
                            value={aggregation.years[0]}
                            onChange={(event) => setAggregation({
                              ...aggregation,
                              years: clampYearRangeToBounds(
                                [Number(event.target.value), aggregation.years?.[1] ?? Number(event.target.value)],
                                aggregationYearBounds
                              )
                            })}
                          />
                        </label>
                        <label>
                          End year
                          <input
                            type="number"
                            min={aggregationYearBounds?.[0]}
                            max={aggregationYearBounds?.[1]}
                            value={aggregation.years[1]}
                            onChange={(event) => setAggregation({
                              ...aggregation,
                              years: clampYearRangeToBounds(
                                [aggregation.years?.[0] ?? Number(event.target.value), Number(event.target.value)],
                                aggregationYearBounds
                              )
                            })}
                          />
                        </label>
                      </div>
                    )}
                    {aggregation.months && (
                      <div className="mini-form-grid">
                        <label>
                          Start month
                          <input type="number" min={1} max={12} value={aggregation.months[0]} onChange={(event) => setAggregation({ ...aggregation, months: [clamp(Number(event.target.value), 1, 12), aggregation.months?.[1] ?? 12] })} />
                        </label>
                        <label>
                          End month
                          <input type="number" min={1} max={12} value={aggregation.months[1]} onChange={(event) => setAggregation({ ...aggregation, months: [aggregation.months?.[0] ?? 1, clamp(Number(event.target.value), 1, 12)] })} />
                        </label>
                      </div>
                    )}
                    <button className="primary" disabled={!canAddAggregation} onClick={addAggregation}>Add aggregation</button>
                    <div className="aggregation-list compact-list">
                      {aggregations.map((item, index) => (
                        <div className="aggregation-chip" key={`${item.name}-${index}`}>
                          <span>
                            <strong>{item.name}</strong>
                            <small>{item.metric}{item.months ? ` · months ${item.months[0]}-${item.months[1]}` : ""}{item.years ? ` · years ${item.years[0]}-${item.years[1]}` : ""}</small>
                          </span>
                          <button className="ghost danger" onClick={() => setAggregations(aggregations.filter((_, itemIndex) => itemIndex !== index))}>Remove</button>
                        </div>
                      ))}
                      {aggregations.length === 0 && (
                        <div className="empty-state">Add at least one aggregation before continuing.</div>
                      )}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="notice info">This source is static; no temporal selection is needed.</div>
            )}
          </div>
        )}

        {step === "resampling" && source && variable && (
          <div className="picker-step-panel temporal-step-grid">
            <div className="dimension-box">
              <strong>Output resampling</strong>
              <ResamplingMethodControl
                catalog={catalog}
                value={resamplingMethod}
                onChange={setResamplingMethod}
                variable={variable}
                isCategoryFraction={selectedCategoryFractions.length > 0}
              />
            </div>
            {sourceResolutionChoices(source).length > 0 && (
              <div className="dimension-box">
                <strong>Original source resolution</strong>
                <label>
                  Resolution used for download/build
                  <select value={sourceResolution} onChange={(event) => setSourceResolution(event.target.value)}>
                    {sourceResolutionChoices(source).map((item) => (
                      <option key={item} value={item}>{item}</option>
                    ))}
                  </select>
                  <small className="field-hint">Choose the native product resolution when the provider offers several variants.</small>
                </label>
              </div>
            )}
            <div className="notice info compact-notice">
              Resampling is applied when source data are aligned to the project target grid. Category fractions
              should normally use average; original categorical codes usually use nearest or mode.
            </div>
          </div>
        )}

        <div className="picker-footer">
          <span className="field-hint">
            {source && variable ? `${sourceShortName(source)} · ${variable.name}` : "Select an input source"}
          </span>
          <div className="button-row">
            {step !== "resampling" && origin === "official" && !waitingForAddedAggregation && (
              <button className="primary" disabled={!canGoNext()} onClick={goNext}>Next</button>
            )}
            {step === "resampling" && (
              <button className="primary" disabled={!canConfirmOfficial} onClick={confirmOfficial}>Confirm input</button>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function OfficialLayersBuilder({
  catalog,
  addFeatures
}: {
  catalog: WorkbenchCatalog;
  addFeatures: (features: DatasetFeatureConfig[]) => void;
}) {
  const [sourceId, setSourceId] = useState(catalog.sources[0]?.id ?? "");
  const source = catalog.sources.find((item) => item.id === sourceId) ?? catalog.sources[0];
  const variables = source ? sourceVariables(source) : [];
  const [selectedVariables, setSelectedVariables] = useState<string[]>([]);
  const [categorySelections, setCategorySelections] = useState<Record<string, string[]>>({});
  const [dimensions, setDimensions] = useState<Record<string, string[]>>({});
  const [temporalMode, setTemporalMode] = useState<FeatureTemporalMode>("static");
  const [temporalLayers, setTemporalLayers] = useState<TemporalSelection["layers"]>({
    annual: false,
    annual_index: false,
    months: [],
    seasons: [],
    years: []
  });
  const [aggregation, setAggregation] = useState<CustomAggregation>(aggregationDefaults(source, [], catalog.supported_metrics[0] ?? "mean"));
  const [aggregations, setAggregations] = useState<CustomAggregation[]>([]);
  const [resamplingByOutput, setResamplingByOutput] = useState<Record<string, string>>({});
  const [sourceResolution, setSourceResolution] = useState(source?.source_resolution ?? sourceResolutionChoices(source)[0] ?? "");

  useEffect(() => {
    if (!source) return;
    const nextDimensions: Record<string, string[]> = {};
    for (const [key] of sourceDimensionEntries(source)) {
      nextDimensions[key] = [];
    }
    setDimensions(nextDimensions);
    setSelectedVariables([]);
    setCategorySelections({});
    setTemporalMode(defaultTemporalModeForSource(source));
    setTemporalLayers(initialTemporalLayers(source));
    setAggregation(aggregationDefaults(source, [], catalog.supported_metrics[0] ?? "mean"));
    setAggregations([]);
    setResamplingByOutput({});
    setSourceResolution(source.source_resolution ?? sourceResolutionChoices(source)[0] ?? "");
  }, [source?.id]);

  const sourceReady = dimensionsAreComplete(source, dimensions)
    && temporalSelectionIsComplete(source, temporalMode, temporalLayers, aggregation.name, aggregation, aggregations, dimensions);
  const aggregationYearBounds = effectiveAggregationYearBounds(source, dimensions);
  const canAddAggregation = aggregationDraftIsComplete(source, dimensions, aggregation)
    && !aggregations.some((item) => item.name === aggregation.name);
  const selectedFractionRequests = variables.flatMap((variable) =>
    (categorySelections[variable.name] ?? []).flatMap((token) => {
      const item = variable.category_classes?.find((candidate) => categoryClassToken(candidate) === token);
      return item ? [{ variable, categoryFraction: categoryFractionConfig(variable, item) }] : [];
    })
  );
  const selectedOutputCount = selectedVariables.length + selectedFractionRequests.length;
  const outputResampling = (key: string, variable: VariableCatalog, isCategoryFraction = false) =>
    resamplingByOutput[key] ?? defaultResamplingForOutput(variable, isCategoryFraction);
  const setOutputResampling = (key: string, method: string) =>
    setResamplingByOutput((current) => ({ ...current, [key]: method }));
  function addAggregation() {
    if (!canAddAggregation) return;
    setAggregations([...aggregations, { ...aggregation }]);
    setAggregation({
      ...aggregation,
      name: `${aggregation.name}_${aggregations.length + 2}`
    });
  }

  function addSelected() {
    if (!source) return;
    const activeAggregations = (temporalMode === "aggregate" || temporalMode === "postprocess_aggregate")
      ? aggregations
      : [aggregation];
    const originalFeatures = selectedVariables.flatMap((name) => {
      const variable = variables.find((item) => item.name === name);
      if (!variable) return [];
      const safeName = sanitizeToken(name);
      const resampling = outputResampling(name, variable, false);
      const outputs = activeAggregations.flatMap((item) => buildSourceLayerOutputs({
          featureName: safeName,
          source,
          variable,
          dimensions,
          temporalMode,
          years: temporalLayers.years,
          layers: temporalLayers,
          aggregation: { ...item, variables: [name] },
          sourceResolution: sourceResolution || source.source_resolution,
          resampling
        }));
      return [{
        name: safeName,
        title: variable.description || humanizeId(name),
        description: variable.description,
        unit: variable.unit ?? undefined,
        value_semantics: variable.value_semantics ?? variable.data_type,
        output_dtype: "float32",
        build_type: "source_layer" as const,
        outputs
      }];
    });
    const fractionFeatures = selectedFractionRequests.map(({ variable, categoryFraction }) => {
      const resampling = outputResampling(categoryFraction.name, variable, true);
      const fractionWithResampling = { ...categoryFraction, resampling };
      const outputs = activeAggregations.flatMap((item) => buildSourceLayerOutputs({
          featureName: categoryFraction.name,
          source,
          variable,
          dimensions,
          temporalMode,
          years: temporalLayers.years,
          layers: temporalLayers,
          aggregation: { ...item, variables: [categoryFraction.variable] },
          categoryFraction: fractionWithResampling,
          sourceResolution: sourceResolution || source.source_resolution,
          resampling
        }));
      return {
        name: categoryFraction.name,
        title: categoryFraction.label ? `${categoryFraction.label} fraction` : humanizeId(categoryFraction.name),
        description: `Fraction of ${variable.description || variable.name} classes: ${categoryFraction.class_values.join(", ")}`,
        unit: "fraction",
        value_semantics: "fraction",
        output_dtype: "float32",
        build_type: "source_layer" as const,
        outputs
      };
    });
    addFeatures([...originalFeatures, ...fractionFeatures]);
    setSelectedVariables([]);
    setCategorySelections({});
  }

  return (
    <section className="panel feature-builder-panel">
      <div className="panel-head">
        <h2>Add official source layers</h2>
        <span className="field-hint">No derived processing</span>
      </div>
      <div className="builder-layout">
        <div className="picker-column">
          <h3>Sources</h3>
          {catalog.sources.map((item) => (
            <button
              key={item.id}
              className={`source-select-button ${item.id === source?.id ? "active" : ""}`}
              onClick={() => setSourceId(item.id)}
            >
              <strong>{sourceDisplayName(item)}</strong>
              <small>{item.id}</small>
            </button>
          ))}
        </div>
        <div className="picker-column">
          <h3>Features</h3>
          {variables.map((variable) => (
            <div key={variable.name} className="official-variable-block">
              <label className="check-row rich">
                <input
                  type="checkbox"
                  checked={selectedVariables.includes(variable.name)}
                  onChange={() => setSelectedVariables(toggleValue(selectedVariables, variable.name))}
                />
                <span>
                  <strong>{variable.description || humanizeId(variable.name)}</strong>
                  <small>{variable.name}{variable.unit ? ` · ${variable.unit}` : ""}</small>
                </span>
              </label>
              {selectedVariables.includes(variable.name) && (
                <div className="resampling-inline">
                  <ResamplingMethodControl
                    catalog={catalog}
                    value={outputResampling(variable.name, variable, false)}
                    onChange={(method) => setOutputResampling(variable.name, method)}
                    variable={variable}
                  />
                </div>
              )}
              {variable.category_classes && variable.category_classes.length > 0 && (
                <div className="category-slice-list">
                  {variable.category_classes.map((item) => {
                    const token = categoryClassToken(item);
                    const selected = categorySelections[variable.name] ?? [];
                    const fraction = categoryFractionConfig(variable, item);
                    return (
                      <div key={token} className="fraction-choice">
                        <label className="check-row rich subtle-check">
                          <input
                            type="checkbox"
                            checked={selected.includes(token)}
                            onChange={() => setCategorySelections({
                              ...categorySelections,
                              [variable.name]: toggleValue(selected, token)
                            })}
                          />
                          <span>
                            <strong>{item.label || item.name || token}</strong>
                            <small>category fraction · {categoryClassValues(item).join(", ")}</small>
                          </span>
                        </label>
                        {selected.includes(token) && (
                          <div className="resampling-inline">
                            <ResamplingMethodControl
                              catalog={catalog}
                              value={outputResampling(fraction.name, variable, true)}
                              onChange={(method) => setOutputResampling(fraction.name, method)}
                              variable={variable}
                              isCategoryFraction
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="picker-column">
          <h3>Dimensions and temporal</h3>
          {source && sourceResolutionChoices(source).length > 0 && (
            <div className="dimension-box">
              <strong>Original source resolution</strong>
              <label>
                Resolution used for download/build
                <select value={sourceResolution} onChange={(event) => setSourceResolution(event.target.value)}>
                  {sourceResolutionChoices(source).map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
                <small className="field-hint">Choose the native product resolution when the provider offers several variants.</small>
              </label>
            </div>
          )}
          {source && sourceDimensionEntries(source).map(([key, values]) => (
            <div className="dimension-box" key={key}>
              <strong>{humanizeId(key)}</strong>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={allValuesSelected(dimensions[key] ?? [], values)}
                  onChange={() => setDimensions({
                    ...dimensions,
                    [key]: allValuesSelected(dimensions[key] ?? [], values) ? [] : values
                  })}
                />
                <span>All</span>
              </label>
              {values.map((value) => (
                <label key={value} className="check-row">
                  <input
                    type="checkbox"
                    checked={(dimensions[key] ?? []).includes(value)}
                    onChange={() => setDimensions({ ...dimensions, [key]: toggleValue(dimensions[key] ?? [], value) })}
                  />
                  <span>{value}</span>
                </label>
              ))}
            </div>
          ))}
          {source?.temporal && (
            <div className="dimension-box">
              <strong>Temporal mode</strong>
              {supportedTemporalModes(source).map((mode) => (
                <label key={mode} className="check-row">
                  <input
                    type="radio"
                    name="official-temporal-mode"
                    checked={temporalMode === mode}
                    onChange={() => setTemporalMode(mode)}
                  />
                  <span>{humanizeId(mode)}</span>
                </label>
              ))}
            </div>
          )}
          {temporalMode === "supplied_layers" && source?.temporal && (
            <div className="dimension-box">
              <strong>Supplied layers</strong>
              {source.temporal.temporal_layers?.annual && (
                <label className="check-row">
                  <input type="checkbox" checked={temporalLayers.annual} onChange={() => setTemporalLayers({ ...temporalLayers, annual: !temporalLayers.annual })} />
                  <span>annual</span>
                </label>
              )}
              {source.temporal.temporal_layers?.annual_index && (
                <label className="check-row">
                  <input type="checkbox" checked={temporalLayers.annual_index} onChange={() => setTemporalLayers({ ...temporalLayers, annual_index: !temporalLayers.annual_index })} />
                  <span>annual index</span>
                </label>
              )}
              {(source.temporal.temporal_layers?.years ?? []).length > 1 && (
                <label className="check-row">
                  <input
                    type="checkbox"
                    checked={allValuesSelected(temporalLayers.years, source.temporal.temporal_layers?.years ?? [])}
                    onChange={() => {
                      const years = source.temporal?.temporal_layers?.years ?? [];
                      setTemporalLayers({
                        ...temporalLayers,
                        years: allValuesSelected(temporalLayers.years, years) ? [] : [...years]
                      });
                    }}
                  />
                  <span>All years</span>
                </label>
              )}
              {(source.temporal.temporal_layers?.years ?? []).map((year) => (
                <label key={year} className="check-row">
                  <input type="checkbox" checked={temporalLayers.years.includes(year)} onChange={() => setTemporalLayers({ ...temporalLayers, years: toggleValue(temporalLayers.years, year) })} />
                  <span>{year}</span>
                </label>
              ))}
              {(source.temporal.temporal_layers?.months ?? []).length > 1 && (
                <label className="check-row">
                  <input
                    type="checkbox"
                    checked={allValuesSelected(temporalLayers.months, source.temporal.temporal_layers?.months ?? [])}
                    onChange={() => {
                      const months = source.temporal?.temporal_layers?.months ?? [];
                      setTemporalLayers({
                        ...temporalLayers,
                        months: allValuesSelected(temporalLayers.months, months) ? [] : [...months]
                      });
                    }}
                  />
                  <span>All months</span>
                </label>
              )}
              {(source.temporal.temporal_layers?.months ?? []).map((month) => (
                <label key={month} className="check-row">
                  <input type="checkbox" checked={temporalLayers.months.includes(month)} onChange={() => setTemporalLayers({ ...temporalLayers, months: toggleValue(temporalLayers.months, month) })} />
                  <span>{month}</span>
                </label>
              ))}
              {(source.temporal.temporal_layers?.seasons ?? []).length > 1 && (
                <label className="check-row">
                  <input
                    type="checkbox"
                    checked={allValuesSelected(temporalLayers.seasons, source.temporal.temporal_layers?.seasons ?? [])}
                    onChange={() => {
                      const seasons = source.temporal?.temporal_layers?.seasons ?? [];
                      setTemporalLayers({
                        ...temporalLayers,
                        seasons: allValuesSelected(temporalLayers.seasons, seasons) ? [] : [...seasons]
                      });
                    }}
                  />
                  <span>All seasons</span>
                </label>
              )}
              {(source.temporal.temporal_layers?.seasons ?? []).map((season) => (
                <label key={season} className="check-row">
                  <input type="checkbox" checked={temporalLayers.seasons.includes(season)} onChange={() => setTemporalLayers({ ...temporalLayers, seasons: toggleValue(temporalLayers.seasons, season) })} />
                  <span>{season}</span>
                </label>
              ))}
            </div>
          )}
          {temporalMode === "raw_slices" && source?.temporal && (
            <div className="dimension-box temporal-aggregate-box">
              <strong>Raw slices</strong>
              <div className="mini-form-grid">
                <label>
                  Start month
                  <input
                    type="number"
                    min={1}
                    max={12}
                    value={aggregation.months?.[0] ?? 1}
                    onChange={(event) => setAggregation({ ...aggregation, months: [clamp(Number(event.target.value), 1, 12), aggregation.months?.[1] ?? 12] })}
                  />
                </label>
                <label>
                  End month
                  <input
                    type="number"
                    min={1}
                    max={12}
                    value={aggregation.months?.[1] ?? 12}
                    onChange={(event) => setAggregation({ ...aggregation, months: [aggregation.months?.[0] ?? 1, clamp(Number(event.target.value), 1, 12)] })}
                  />
                </label>
              </div>
              {source.temporal.kind === "year_month_series" && (
                <div className="mini-form-grid">
                  <label>
                    Start year
                      <input
                        type="number"
                        min={aggregationYearBounds?.[0]}
                        max={aggregationYearBounds?.[1]}
                        value={aggregation.years?.[0] ?? source.temporal.default_years?.[0] ?? ""}
                        onChange={(event) => setAggregation({
                          ...aggregation,
                          years: clampYearRangeToBounds(
                            [Number(event.target.value), aggregation.years?.[1] ?? Number(event.target.value)],
                            aggregationYearBounds
                          )
                        })}
                      />
                    </label>
                    <label>
                      End year
                      <input
                        type="number"
                        min={aggregationYearBounds?.[0]}
                        max={aggregationYearBounds?.[1]}
                        value={aggregation.years?.[1] ?? source.temporal.default_years?.[1] ?? ""}
                        onChange={(event) => setAggregation({
                          ...aggregation,
                          years: clampYearRangeToBounds(
                            [aggregation.years?.[0] ?? Number(event.target.value), Number(event.target.value)],
                            aggregationYearBounds
                          )
                        })}
                      />
                    </label>
                </div>
              )}
            </div>
          )}
          {(temporalMode === "aggregate" || temporalMode === "postprocess_aggregate") && (
            <div className="dimension-box temporal-aggregate-box">
              <strong>Aggregation</strong>
              {aggregationYearBounds && (
                <div className="notice info compact-notice">
                  Available year range after selected dimensions: {aggregationYearBounds[0]}-{aggregationYearBounds[1]}.
                </div>
              )}
              <label>
                Name
                <input value={aggregation.name} onChange={(event) => setAggregation({ ...aggregation, name: sanitizeToken(event.target.value) })} />
              </label>
              <label>
                Metric
                <select value={aggregation.metric} onChange={(event) => setAggregation({ ...aggregation, metric: event.target.value })}>
                  {catalog.supported_metrics.map((metric) => (
                    <option key={metric} value={metric}>{metric}</option>
                  ))}
                </select>
              </label>
              {aggregation.years && (
                <div className="mini-form-grid">
                  <label>
                    Start year
                    <input
                      type="number"
                      min={aggregationYearBounds?.[0]}
                      max={aggregationYearBounds?.[1]}
                      value={aggregation.years[0]}
                      onChange={(event) => setAggregation({
                        ...aggregation,
                        years: clampYearRangeToBounds(
                          [Number(event.target.value), aggregation.years?.[1] ?? Number(event.target.value)],
                          aggregationYearBounds
                        )
                      })}
                    />
                  </label>
                  <label>
                    End year
                    <input
                      type="number"
                      min={aggregationYearBounds?.[0]}
                      max={aggregationYearBounds?.[1]}
                      value={aggregation.years[1]}
                      onChange={(event) => setAggregation({
                        ...aggregation,
                        years: clampYearRangeToBounds(
                          [aggregation.years?.[0] ?? Number(event.target.value), Number(event.target.value)],
                          aggregationYearBounds
                        )
                      })}
                    />
                  </label>
                </div>
              )}
              {aggregation.months && (
                <div className="mini-form-grid">
                  <label>
                    Start month
                    <input
                      type="number"
                      min={1}
                      max={12}
                      value={aggregation.months[0]}
                      onChange={(event) => setAggregation({ ...aggregation, months: [clamp(Number(event.target.value), 1, 12), aggregation.months?.[1] ?? 12] })}
                    />
                  </label>
                  <label>
                    End month
                    <input
                      type="number"
                      min={1}
                      max={12}
                      value={aggregation.months[1]}
                      onChange={(event) => setAggregation({ ...aggregation, months: [aggregation.months?.[0] ?? 1, clamp(Number(event.target.value), 1, 12)] })}
                    />
                  </label>
                </div>
              )}
              <button className="primary" disabled={!canAddAggregation} onClick={addAggregation}>Add aggregation</button>
              <div className="aggregation-list compact-list">
                {aggregations.map((item, index) => (
                  <div className="aggregation-chip" key={`${item.name}-${index}`}>
                    <span>
                      <strong>{item.name}</strong>
                      <small>{item.metric}{item.months ? ` · months ${item.months[0]}-${item.months[1]}` : ""}{item.years ? ` · years ${item.years[0]}-${item.years[1]}` : ""}</small>
                    </span>
                    <button className="ghost danger" onClick={() => setAggregations(aggregations.filter((_, itemIndex) => itemIndex !== index))}>Remove</button>
                  </div>
                ))}
                {aggregations.length === 0 && (
                  <div className="empty-state">Add at least one aggregation before adding features.</div>
                )}
              </div>
            </div>
          )}
          <button className="primary" disabled={selectedOutputCount === 0 || !sourceReady} onClick={addSelected}>
            Add {selectedOutputCount || ""} feature{selectedOutputCount === 1 ? "" : "s"}
          </button>
        </div>
      </div>
    </section>
  );
}

function CustomFeatureBuilder({
  catalog,
  existingFeatures,
  projectName,
  addFeature,
  editing,
  clearEditing
}: {
  catalog: WorkbenchCatalog;
  existingFeatures: DatasetFeatureConfig[];
  projectName: string;
  addFeature: (feature: DatasetFeatureConfig) => void;
  editing?: DatasetFeatureConfig | null;
  clearEditing?: () => void;
}) {
  const bundlesFromInputs = (items?: Record<string, DatasetFeatureInput>): Record<string, InputBundle> => Object.fromEntries(
    Object.entries(items ?? {}).map(([alias, input]) => [
      alias,
      {
        label: input.kind === "feature" ? `${projectName} · ${input.output || input.feature}` : `${input.source_id} · ${input.variable || input.layer || "layer"}`,
        outputs: [{
          name: input.kind === "feature" ? input.output || input.feature : input.variable || input.layer || alias,
          label: input.kind === "feature" ? input.output || input.feature : `${input.source_id} · ${input.variable || input.layer || "layer"}`,
          input,
          suffix: input.kind === "feature" ? input.output || input.feature : input.variable || input.layer
        }]
      }
    ])
  );
  const [name, setName] = useState(editing?.name ?? "custom_feature");
  const [title, setTitle] = useState(editing?.title ?? "");
  const [description, setDescription] = useState(editing?.description ?? "");
  const [unit, setUnit] = useState(editing?.unit ?? "");
  const [semantics, setSemantics] = useState(editing?.value_semantics ?? "intensive");
  const [buildType, setBuildType] = useState<DatasetFeatureConfig["build_type"]>(editing?.build_type ?? "expression");
  const [operation, setOperation] = useState(editing?.operation ?? "terrain");
	  const [method, setMethod] = useState(editing?.method ?? "slope");
	  const [recipe, setRecipe] = useState(editing?.recipe ?? "thermal_range");
	  const [expression, setExpression] = useState(editing?.expression ?? "x");
	  const [parametersText, setParametersText] = useState(JSON.stringify(editing?.parameters ?? {}, null, 2));
	  const [inputBundles, setInputBundles] = useState<Record<string, InputBundle>>(bundlesFromInputs(editing?.inputs));
	  const [pickingAlias, setPickingAlias] = useState<string | null>(null);
	  const [thresholdValue, setThresholdValue] = useState<string>(
	    editing?.parameters?.threshold !== undefined ? String(editing.parameters.threshold) : ""
	  );
	  const [classValue, setClassValue] = useState<string>(
	    editing?.parameters?.class_value !== undefined ? String(editing.parameters.class_value) : ""
	  );
	  const [radius, setRadius] = useState<number>(
	    Number(editing?.parameters?.radius ?? 1)
	  );
	  const [reclassText, setReclassText] = useState(
	    JSON.stringify(editing?.parameters?.classes ?? {}, null, 2)
	  );
	  const terrainMethods = ["slope", "aspect", "ruggedness", "tpi", "roughness"];
	  const focalMethods = ["mean", "std", "min", "max", "sum", "majority", "diversity"];
	  const distanceMethods = ["distance_to_mask", "distance_to_class"];
	  const guidedRecipeOptions = [
	    ["thermal_range", "Thermal range", "Maximum temperature minus minimum temperature."],
	    ["water_balance", "Water balance", "Precipitation minus potential evapotranspiration."],
	    ["aridity_index", "Aridity index", "Precipitation divided by PET."],
	    ["snow_persistence_ratio", "Snow persistence ratio", "Snow-day count divided by valid observation count for the same period."],
	    ["seasonal_contrast", "Seasonal contrast", "Contrast between two numeric seasonal or period layers."]
	  ];
	  const maskingOptions = [
	    ["binary_threshold_mask", "Binary threshold mask", "Numeric input converted to a final-grid 0/1 mask with a threshold."],
	    ["class_mask", "Final-grid class mask", "Categorical input converted to 0/1 after source resampling."],
	    ["reclassification", "Reclassification", "Map selected class codes to new numeric/categorical values."]
	  ];
	  const spatialOptions = [
	    ["terrain", "DEM terrain", "Terrain derivatives from a DEM/elevation layer."],
	    ["focal", "Focal window", "Neighbourhood statistics around each cell."],
	    ["distance", "Distance", "Distance to positive mask pixels or a selected class."]
	  ];

  useEffect(() => {
    if (!editing) return;
    setName(editing.name);
    setTitle(editing.title ?? "");
    setDescription(editing.description ?? "");
    setUnit(editing.unit ?? "");
    setSemantics(editing.value_semantics ?? "intensive");
    setBuildType(editing.build_type);
    setOperation(editing.operation ?? "terrain");
    setMethod(editing.method ?? "slope");
	    setRecipe(editing.recipe ?? "thermal_range");
	    setExpression(editing.expression ?? "x");
	    setParametersText(JSON.stringify(editing.parameters ?? {}, null, 2));
	    setInputBundles(bundlesFromInputs(editing.inputs));
	    setThresholdValue(editing.parameters?.threshold !== undefined ? String(editing.parameters.threshold) : "");
	    setClassValue(editing.parameters?.class_value !== undefined ? String(editing.parameters.class_value) : "");
	    setRadius(Number(editing.parameters?.radius ?? 1));
	    setReclassText(JSON.stringify(editing.parameters?.classes ?? {}, null, 2));
	  }, [editing]);

	  useEffect(() => {
	    if (buildType === "recipe" && !guidedRecipeOptions.some(([value]) => value === recipe)) {
	      setRecipe("thermal_range");
	    }
	    if (buildType === "masking" && !maskingOptions.some(([value]) => value === recipe)) {
	      setRecipe("binary_threshold_mask");
	    }
	  }, [buildType, recipe]);

	  useEffect(() => {
	    if (operation === "terrain" && !terrainMethods.includes(method)) setMethod("slope");
	    if (operation === "focal" && !focalMethods.includes(method)) setMethod("mean");
	    if (operation === "distance" && !distanceMethods.includes(method)) setMethod("distance_to_mask");
	  }, [operation]);

	  const aliases = buildType === "recipe"
	    ? recipe === "thermal_range"
	      ? ["tmax", "tmin"]
	      : recipe === "water_balance" || recipe === "aridity_index"
	        ? ["prec", "pet"]
	        : recipe === "snow_persistence_ratio"
	          ? ["snow_days", "valid_days"]
	          : recipe === "seasonal_contrast"
	            ? ["a", "b"]
	            : ["x"]
	    : buildType === "spatial"
	      ? [operation === "terrain" ? "dem" : operation === "distance" ? "mask" : "x"]
	      : buildType === "expression"
	        ? ["x", "y", "z"]
	        : ["x"];
	  const aliasesKey = aliases.join("|");

	  useEffect(() => {
	    setInputBundles((current) => Object.fromEntries(
	      Object.entries(current).filter(([alias]) => aliases.includes(alias))
	    ));
	  }, [aliasesKey]);

	  function variableFilter(variable: VariableCatalog, alias?: string) {
	    const haystack = `${variable.name} ${variable.description ?? ""} ${variable.value_semantics ?? ""} ${variable.data_type ?? ""}`;
    if (buildType === "spatial" && operation === "terrain") {
      return /\b(dem|elev|elevation|altitude|height)\b/i.test(haystack);
    }
    if (buildType === "spatial" && operation === "focal") {
      return !/\b(categorical|ordinal|class)\b/i.test(haystack);
    }
    if (buildType === "spatial" && operation === "distance") {
      return /\b(categorical|binary|ordinal|class|mask|presence|landcover|road|track|building|settlement)\b/i.test(
        haystack
      );
    }
    if (buildType === "recipe" && recipe === "thermal_range") {
      if (alias === "tmax") return /\b(tmax|tasmax|max(?:imum)? temperature)\b/i.test(haystack);
      if (alias === "tmin") return /\b(tmin|tasmin|min(?:imum)? temperature)\b/i.test(haystack);
      return /\b(tmin|tmax|tasmin|tasmax|temperature)\b/i.test(haystack);
    }
	    if (buildType === "recipe" && (recipe === "water_balance" || recipe === "aridity_index")) {
	      if (alias === "prec") return /\b(prec|precip|precipitation|rain)\b/i.test(haystack);
	      if (alias === "pet") return /\b(pet|evapo|evapotranspiration)\b/i.test(haystack);
	    }
	    if (buildType === "recipe" && recipe === "snow_persistence_ratio") {
	      if (alias === "snow_days") return /\b(snow|nieve|neu|snow_days|snow_days_count|scd|snow_cover)\b/i.test(haystack);
	      if (alias === "valid_days") return /\b(valid|observation|observations|valid_days|valid_observations|count)\b/i.test(haystack);
	    }
	    if (buildType === "recipe" && recipe === "seasonal_contrast") {
	      return !/\b(categorical|ordinal|class|mask)\b/i.test(haystack);
	    }
	    if (buildType === "masking" && recipe === "binary_threshold_mask") {
	      return !/\b(categorical|ordinal|class)\b/i.test(haystack);
	    }
    if (buildType === "masking" && (recipe === "class_mask" || recipe === "reclassification")) {
      return /\b(categorical|ordinal|class|landcover|settlement|mask)\b/i.test(haystack);
	    }
	    return true;
	  }

	  function inputVariable(alias: string) {
	    return inputBundles[alias]?.outputs[0]?.variable;
	  }

	  function requiredAliases() {
	    if (buildType === "expression") {
	      return aliases.filter((alias) => alias === "x" || new RegExp(`\\b${alias}\\b`).test(expression));
	    }
	    return aliases;
	  }

	  function classOptionsForAlias(alias: string) {
	    return inputVariable(alias)?.category_classes ?? [];
	  }

	  function thresholdRange() {
	    const range = inputVariable("x")?.valid_range;
	    return Array.isArray(range) && range.length >= 2 ? [Number(range[0]), Number(range[1])] as [number, number] : undefined;
	  }

	  function thresholdIsValid() {
	    if (!(buildType === "masking" && recipe === "binary_threshold_mask")) return true;
	    if (thresholdValue.trim() === "") return false;
	    const value = Number(thresholdValue);
	    if (!Number.isFinite(value)) return false;
	    const range = thresholdRange();
	    return !range || (value >= range[0] && value <= range[1]);
	  }

	  function classValueIsValid() {
	    if (buildType === "masking" && recipe === "class_mask") return classValue.trim().length > 0;
	    if (buildType === "spatial" && operation === "distance" && method === "distance_to_class") {
	      return classValue.trim().length > 0;
	    }
	    return true;
	  }

	  function reclassificationIsValid() {
	    if (!(buildType === "masking" && recipe === "reclassification")) return true;
	    try {
	      const parsed = JSON.parse(reclassText);
	      return Boolean(parsed) && typeof parsed === "object" && !Array.isArray(parsed) && Object.keys(parsed).length > 0;
	    } catch {
	      return false;
	    }
	  }

	  function radiusIsRequired() {
	    return buildType === "spatial" && (
	      operation === "focal" ||
	      (operation === "terrain" && ["ruggedness", "tpi", "roughness"].includes(method))
	    );
	  }

	  function radiusIsValid() {
	    return !radiusIsRequired() || (Number.isFinite(radius) && radius >= 1);
	  }

	  function canAddFeature() {
	    const required = requiredAliases();
	    return Boolean(name.trim())
	      && required.every((alias) => inputBundles[alias]?.outputs.length > 0)
	      && thresholdIsValid()
	      && classValueIsValid()
	      && reclassificationIsValid()
	      && radiusIsValid();
	  }

	  function operationParameters(extra: Record<string, unknown>) {
	    const parameters = { ...extra };
	    if (buildType === "masking" && recipe === "binary_threshold_mask") {
	      parameters.operator = ">=";
	      parameters.threshold = Number(thresholdValue);
	    }
	    if (buildType === "masking" && recipe === "class_mask") {
	      parameters.class_value = Number.isNaN(Number(classValue)) ? classValue : Number(classValue);
	    }
	    if (buildType === "masking" && recipe === "reclassification") {
	      parameters.classes = JSON.parse(reclassText);
	    }
	    if (buildType === "spatial" && operation === "distance" && method === "distance_to_class") {
	      parameters.class_value = Number.isNaN(Number(classValue)) ? classValue : Number(classValue);
	    }
	    if (radiusIsRequired()) {
	      parameters.radius = Math.max(1, Math.round(radius));
	    }
	    return parameters;
	  }

	  function add() {
	    let extraParameters: Record<string, unknown> = {};
	    try {
	      extraParameters = buildType !== "source_layer" && parametersText.trim() ? JSON.parse(parametersText) : {};
	    } catch {
	      window.alert("Parameters must be valid JSON.");
	      return;
	    }
	    const safeName = sanitizeToken(name);
	    if (!canAddFeature()) {
	      window.alert("Complete all required inputs before adding the feature.");
	      return;
	    }
	    const parameters = operationParameters(extraParameters);
	    const activeAliases = requiredAliases();
	    const bundles = activeAliases.map((alias) => ({ alias, bundle: inputBundles[alias] }));
    const temporalBundles = bundles.filter(({ bundle }) => bundle.outputs.some((output) => output.temporalKey));
    let combinations: Array<Array<{ alias: string; option: InputOutputOption }>> = [];

    if (temporalBundles.length >= 2) {
      const commonTemporalKeys = temporalBundles
        .map(({ bundle }) => new Set(bundle.outputs.map((output) => output.temporalKey).filter(Boolean) as string[]))
        .reduce<Set<string> | null>((current, next) => {
          if (current === null) return next;
          return new Set([...current].filter((key) => next.has(key)));
        }, null);
      const keys = [...(commonTemporalKeys ?? new Set<string>())];
      if (keys.length === 0) {
        window.alert("These inputs do not share temporal outputs. Choose aligned temporal selections.");
        return;
      }
      combinations = keys.map((key) => bundles.map(({ alias, bundle }) => ({
        alias,
        option: bundle.outputs.find((output) => output.temporalKey === key) ?? bundle.outputs[0]
      })));
    } else {
      combinations = cartesianProduct(bundles.map(({ alias, bundle }) =>
        bundle.outputs.map((option) => ({ alias, option }))
      ));
    }

    const outputs = combinations.map((combo, index) => {
      const suffix = [...new Set(combo.map(({ option }) => option.temporalKey || option.dimensionKey || option.suffix).filter(Boolean))].join("_");
      return {
        name: suffix ? `${safeName}_${sanitizeToken(suffix)}` : combinations.length > 1 ? `${safeName}_${index + 1}` : safeName,
        suffix,
        inputs: Object.fromEntries(combo.map(({ alias, option }) => [alias, option.input])),
        expression: buildType === "expression" ? expression : undefined,
        recipe: buildType === "recipe" || buildType === "masking" ? recipe : undefined,
        operation: buildType === "spatial" ? operation : undefined,
        method: buildType === "spatial" ? method : undefined,
        parameters
      };
    });
    const firstInputs = outputs[0]?.inputs ?? {};
    const feature: DatasetFeatureConfig = {
      name: safeName,
      title: title || humanizeId(safeName),
      description,
	      unit: unit || undefined,
	      value_semantics: buildType === "masking" && recipe !== "reclassification" ? "binary" : semantics,
	      output_dtype: buildType === "masking" && recipe !== "reclassification" ? "uint8" : "float32",
      build_type: buildType,
      inputs: firstInputs,
      expression: buildType === "expression" ? expression : undefined,
      recipe: buildType === "recipe" || buildType === "masking" ? recipe : undefined,
      operation: buildType === "spatial" ? operation : undefined,
      method: buildType === "spatial" ? method : undefined,
      parameters,
      outputs
    };
    addFeature(feature);
    clearEditing?.();
    setName("custom_feature");
	    setTitle("");
	    setDescription("");
	    setInputBundles({});
	    setThresholdValue("");
	    setClassValue("");
	    setRadius(1);
	    setReclassText("{}");
	  }

	  function aliasLabel(alias: string) {
	    const labels: Record<string, string> = {
	      x: "Input x",
	      y: "Input y",
	      z: "Input z",
	      tmax: "Maximum temperature",
	      tmin: "Minimum temperature",
	      prec: "Precipitation",
	      pet: "Potential evapotranspiration",
	      snow_days: "Snow-days layer",
	      valid_days: "Valid-days layer",
	      a: "First numeric layer",
	      b: "Second numeric layer",
	      dem: "DEM input",
	      mask: "Mask or categorical input"
	    };
	    return labels[alias] ?? humanizeId(alias);
	  }

	  function aliasHint(alias: string) {
	    if (buildType === "source_layer") return "Official source layer, temporal slice, aggregation or category fraction.";
	    if (buildType === "expression") return alias === "x"
	      ? "Required base input for the expression."
	      : "Optional; required only if the expression uses this symbol.";
	    if (buildType === "spatial" && operation === "terrain") return "Only DEM/elevation layers are accepted.";
	    if (buildType === "spatial" && operation === "distance") return "Use a binary mask, categorical layer or class-derived mask.";
	    if (buildType === "spatial" && operation === "focal") return "Use a numeric layer for focal statistics.";
	    if (buildType === "masking" && recipe === "binary_threshold_mask") return "Numeric input required; choose a threshold below.";
	    if (buildType === "masking" && recipe === "class_mask") return "Categorical input required. This creates a final-grid 0/1 mask, not a coverage fraction.";
	    if (buildType === "masking" && recipe === "reclassification") return "Categorical/ordinal input required. Provide a class-code mapping below.";
	    if (buildType === "recipe" && recipe === "snow_persistence_ratio") return alias === "snow_days"
	      ? "Snow-day count for the selected period, used as numerator."
	      : "Valid observation count for the same period, used as denominator. Example: a layer counting cloud-free or otherwise valid HRSI snow observations.";
	    if (buildType === "recipe" && recipe === "seasonal_contrast") return "Numeric layer; temporal overlap is kept when both inputs are temporal.";
	    return "Filtered to variables compatible with this operation.";
	  }

	  function renderInputSlot(alias: string, required = true) {
	    return (
	      <div className="input-slot feature-input-slot" key={alias}>
	        <div>
	          <strong>{aliasLabel(alias)}{required ? " *" : ""}</strong>
	          <small>{aliasHint(alias)}</small>
	        </div>
	        <span>{inputBundles[alias]?.label || "No input selected"}</span>
	        <button onClick={() => setPickingAlias(alias)}>Select input</button>
	      </div>
	    );
	  }

	  const required = new Set(requiredAliases());
	  const range = thresholdRange();
	  const classOptions = classOptionsForAlias(buildType === "spatial" ? "mask" : "x");

  return (
    <section className="panel feature-builder-panel custom-feature-builder">
      <div className="panel-head">
        <h2>{editing ? "Edit custom feature" : "Build custom feature"}</h2>
        {editing && <button className="ghost" onClick={clearEditing}>Cancel edit</button>}
      </div>

      <div className="form-grid custom-feature-metadata">
        <label>
          Feature name *
          <input value={name} onChange={(event) => setName(event.target.value)} />
          <small className="field-hint">Stable output name used in the YAML and final raster filenames.</small>
        </label>
        <label>
          Title
          <input value={title} onChange={(event) => setTitle(event.target.value)} />
          <small className="field-hint">Readable title for review and metadata.</small>
        </label>
        <label>
          Unit
          <input value={unit} onChange={(event) => setUnit(event.target.value)} />
          <small className="field-hint">Leave empty if unitless or inherited from the source.</small>
        </label>
        <label>
          Value semantics
          <select value={semantics} onChange={(event) => setSemantics(event.target.value)}>
            {(catalog.value_semantics ?? ["intensive", "percentage", "fraction", "categorical", "binary", "ratio"]).map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
          <small className="field-hint">How the final raster values should be interpreted.</small>
        </label>
        <label className="span-4">
          Description
          <textarea value={description} rows={2} onChange={(event) => setDescription(event.target.value)} />
        </label>
      </div>

      <div className="feature-option-row build-type-row" aria-label="Build type">
        {[
          ["source_layer", "Use official source layer"],
          ["recipe", "Guided recipe"],
          ["masking", "Masking"],
          ["spatial", "Spatial operation"],
          ["expression", "Advanced expression"]
        ].map(([value, label]) => (
          <button
            key={value}
            className={`operation-card compact ${buildType === value ? "active" : ""}`}
            onClick={() => setBuildType(value as DatasetFeatureConfig["build_type"])}
          >
            {label}
          </button>
        ))}
      </div>

      {buildType === "recipe" && (
        <div className="feature-option-row" aria-label="Guided recipe">
          {guidedRecipeOptions.map(([value, label, help]) => (
            <button key={value} className={`operation-card compact ${recipe === value ? "active" : ""}`} onClick={() => setRecipe(value)}>
              <strong>{label}</strong>
              <small>{help}</small>
            </button>
          ))}
        </div>
      )}

      {buildType === "masking" && (
        <div className="feature-option-row three-options" aria-label="Masking type">
          {maskingOptions.map(([value, label, help]) => (
            <button key={value} className={`operation-card compact ${recipe === value ? "active" : ""}`} onClick={() => setRecipe(value)}>
              <strong>{label}</strong>
              <small>{help}</small>
            </button>
          ))}
        </div>
      )}

      {buildType === "spatial" && (
        <>
          <div className="feature-option-row three-options" aria-label="Spatial operation">
            {spatialOptions.map(([value, label, help]) => (
              <button key={value} className={`operation-card compact ${operation === value ? "active" : ""}`} onClick={() => setOperation(value)}>
                <strong>{label}</strong>
                <small>{help}</small>
              </button>
            ))}
          </div>
          <div className="feature-option-row method-row" aria-label="Spatial method">
            {(operation === "terrain" ? terrainMethods : operation === "focal" ? focalMethods : distanceMethods).map((item) => (
              <button key={item} className={`operation-card compact method-card ${method === item ? "active" : ""}`} onClick={() => setMethod(item)}>
                {humanizeId(item)}
              </button>
            ))}
          </div>
        </>
      )}

      <section className="feature-work-area">
        {buildType === "expression" && (
          <>
            <label className="full-row">
              Expression *
              <input value={expression} onChange={(event) => setExpression(event.target.value)} />
              <small className="field-hint">Use x, y and z as selected input layers. Inputs y/z become required when the expression references them.</small>
              <div className="expression-keypad">
                {["where(", "maximum(", "minimum(", "log10(", "sqrt(", "nan", "+", "-", "*", "/"].map((token) => (
                  <button key={token} className="ghost" onClick={() => setExpression(`${expression}${token}`)}>{token}</button>
                ))}
              </div>
            </label>
            <div className="input-grid three-inputs">
              {aliases.map((alias) => renderInputSlot(alias, required.has(alias)))}
            </div>
          </>
        )}

        {buildType === "source_layer" && (
          <div className="input-grid single-input">
            {renderInputSlot("x", true)}
          </div>
        )}

        {buildType === "recipe" && (
          <div className={`input-grid ${aliases.length === 1 ? "single-input" : "two-inputs"}`}>
            {aliases.map((alias) => renderInputSlot(alias, true))}
          </div>
        )}

        {buildType === "masking" && (
          <>
            <div className="notice info compact-notice full-row">
              Use masking for binary final-grid rasters. For habitat percentages from categorical maps, choose
              category fractions in the source picker instead: they are computed before resampling and can become
              true target-cell coverage ratios with average resampling.
            </div>
            <div className="input-grid single-input">
              {renderInputSlot("x", true)}
            </div>
            {recipe === "binary_threshold_mask" && (
              <div className="parameter-grid">
                <label>
                  Threshold *
                  <input type="number" value={thresholdValue} onChange={(event) => setThresholdValue(event.target.value)} />
                  <small className={`field-hint ${thresholdIsValid() ? "" : "error-text"}`}>
                    {range ? `Accepted range for selected variable: ${range[0]} to ${range[1]}.` : "Choose a numeric value that makes sense for the selected input."}
                  </small>
                </label>
              </div>
            )}
            {recipe === "class_mask" && (
              <div className="parameter-grid">
                <label>
                  Class value *
                  {classOptions.length > 0 ? (
                    <select value={classValue} onChange={(event) => setClassValue(event.target.value)}>
                      <option value="">Select class</option>
                      {classOptions.map((item) => {
                        const value = item.value ?? item.values?.[0] ?? item.name ?? item.label ?? "";
                        return <option key={String(value)} value={String(value)}>{item.label || item.name || String(value)}</option>;
                      })}
                    </select>
                  ) : (
                    <input value={classValue} onChange={(event) => setClassValue(event.target.value)} placeholder="class code" />
                  )}
                  <small className="field-hint">This is not a category fraction. It tests the already aligned target-grid layer and returns 1 where the final cell equals this class.</small>
                </label>
              </div>
            )}
            {recipe === "reclassification" && (
              <label className="full-row optional-json-field">
                Class mapping JSON *
                <textarea value={reclassText} rows={5} onChange={(event) => setReclassText(event.target.value)} />
                <small className={`field-hint ${reclassificationIsValid() ? "" : "error-text"}`}>
                  Map original class codes to new values, for example {"{\"10\": 1, \"20\": 2}"}. Unlisted classes keep their original value.
                </small>
              </label>
            )}
          </>
        )}

        {buildType === "spatial" && (
          <>
            <div className="input-grid single-input">
              {renderInputSlot(aliases[0], true)}
            </div>
            {(radiusIsRequired() || (operation === "distance" && method === "distance_to_class")) && (
              <div className="parameter-grid">
                {radiusIsRequired() && (
                  <label>
                    Radius in cells *
                    <input type="number" min={1} value={radius} onChange={(event) => setRadius(Math.max(1, Number(event.target.value)))} />
                    <small className="field-hint">Used by focal windows and terrain neighbourhood metrics such as TPI, ruggedness and roughness.</small>
                  </label>
                )}
                {operation === "distance" && method === "distance_to_class" && (
                  <label>
                    Class value *
                    {classOptions.length > 0 ? (
                      <select value={classValue} onChange={(event) => setClassValue(event.target.value)}>
                        <option value="">Select class</option>
                        {classOptions.map((item) => {
                          const value = item.value ?? item.values?.[0] ?? item.name ?? item.label ?? "";
                          return <option key={String(value)} value={String(value)}>{item.label || item.name || String(value)}</option>;
                        })}
                      </select>
                    ) : (
                      <input value={classValue} onChange={(event) => setClassValue(event.target.value)} placeholder="class code" />
                    )}
                    <small className="field-hint">Distance is computed to cells equal to this class value.</small>
                  </label>
                )}
              </div>
            )}
          </>
        )}

        {buildType !== "source_layer" && (
          <label className="full-row optional-json-field">
            Parameters JSON
            <textarea value={parametersText} rows={3} onChange={(event) => setParametersText(event.target.value)} />
            <small className="field-hint">Optional advanced overrides for the derived operation. Most users can leave this as {"{}"}; required parameters above are filled automatically.</small>
          </label>
        )}
      </section>

      <button className="primary" disabled={!canAddFeature()} onClick={add}>
        {editing ? "Update feature" : "Add final feature"}
      </button>

      {pickingAlias && (
        <FeatureInputPicker
          catalog={catalog}
          existingFeatures={existingFeatures}
          projectName={projectName}
          allowCategoryFractions={!(buildType === "masking" && recipe === "class_mask")}
          filter={(variable) => variableFilter(variable, pickingAlias ?? undefined)}
          onCancel={() => setPickingAlias(null)}
          onConfirm={(bundle) => {
            setInputBundles({ ...inputBundles, [pickingAlias]: bundle });
            setPickingAlias(null);
          }}
        />
      )}
    </section>
  );
}

function FeatureBuilderPanel({
  catalog,
  features,
  setFeatures,
  projectName,
  onReview
}: {
  catalog: WorkbenchCatalog;
  features: DatasetFeatureConfig[];
  setFeatures: (features: DatasetFeatureConfig[]) => void;
  projectName: string;
  onReview: () => void;
}) {
  const [mode, setMode] = useState<"home" | "official" | "custom">("home");
  const [editing, setEditing] = useState<{ feature: DatasetFeatureConfig; index: number } | null>(null);

  function upsertFeature(feature: DatasetFeatureConfig) {
    if (editing) {
      setFeatures(features.map((item, index) => index === editing.index ? feature : item));
      setEditing(null);
      return;
    }
    setFeatures([...features, feature]);
  }

  return (
    <main className="feature-workspace">
      <section className="feature-builder-main">
        <section className="panel feature-mode-panel">
          <div className="panel-head">
            <h2>Feature builder</h2>
            <div className="button-row">
              <button className={mode === "home" ? "primary" : "ghost"} onClick={() => setMode("home")}>Tools</button>
              <button className="primary" disabled={features.length === 0} onClick={onReview}>Review</button>
            </div>
          </div>
          {mode === "home" && (
            <div className="feature-tool-grid">
              <button className="feature-tool-card" onClick={() => setMode("custom")}>
                <strong>Build custom feature</strong>
                <small>Create one final feature with source inputs, derived operations or expressions.</small>
              </button>
              <button className="feature-tool-card" onClick={() => setMode("official")}>
                <strong>Add official source layers</strong>
                <small>Select several original layers and optional dimensions/years without derived processing.</small>
              </button>
            </div>
          )}
        </section>

        {mode === "official" && (
          <OfficialLayersBuilder
            catalog={catalog}
            addFeatures={(items) => setFeatures([...features, ...items])}
          />
        )}
        {mode === "custom" && (
          <CustomFeatureBuilder
            catalog={catalog}
            existingFeatures={features}
            projectName={projectName}
            editing={editing?.feature ?? null}
            clearEditing={() => setEditing(null)}
            addFeature={upsertFeature}
          />
        )}
      </section>
      <FeatureSidebar
        features={features}
        setFeatures={setFeatures}
        onEdit={(feature, index) => {
          setEditing({ feature, index });
          setMode("custom");
        }}
      />
    </main>
  );
}

function FeatureReviewPanel({
  yamlText,
  validation,
  apiError,
  saveStatus,
  validate,
  renderFromServer,
  copyYaml,
  saveYamlToRuns,
  downloadYaml,
  features,
  setFeatures
}: {
  yamlText: string;
  validation: ValidationReport | null;
  apiError: string | null;
  saveStatus: string | null;
  validate: () => void;
  renderFromServer: () => void;
  copyYaml: () => void;
  saveYamlToRuns: () => void;
  downloadYaml: () => void;
  features: DatasetFeatureConfig[];
  setFeatures: (features: DatasetFeatureConfig[]) => void;
}) {
  const expandedOutputs = features.flatMap((feature) =>
    featureOutputNames(feature).map((output) => ({ feature, output }))
  );

  function removeFeature(index: number) {
    const feature = features[index];
    const outputs = new Set(featureOutputNames(feature));
    const dependents = features
      .filter((candidate, candidateIndex) => candidateIndex !== index)
      .filter((candidate) =>
        Object.values(candidate.inputs ?? {}).some((input) =>
          input.kind === "feature" && outputs.has(input.output || input.feature)
        )
      )
      .map((candidate) => candidate.name);
    if (dependents.length > 0) {
      const ok = window.confirm(
        `Removing ${feature.name} will also remove dependent features: ${dependents.join(", ")}. Continue?`
      );
      if (!ok) return;
      setFeatures(features.filter((candidate, candidateIndex) =>
        candidateIndex !== index && !dependents.includes(candidate.name)
      ));
      return;
    }
    setFeatures(features.filter((_, candidateIndex) => candidateIndex !== index));
  }

  return (
    <main className="workspace review-grid feature-review-grid">
      <section className="panel">
        <div className="panel-head">
          <h2>Review final dataset</h2>
          <div className="button-row">
            <button className="primary" onClick={validate}>Validate</button>
            <button onClick={renderFromServer}>Render</button>
            <button onClick={copyYaml}>Copy YAML</button>
            <button onClick={saveYamlToRuns}>Save YAML</button>
            <button onClick={downloadYaml}>Download YAML</button>
          </div>
        </div>
        {apiError && <div className="notice error">{apiError}</div>}
        {saveStatus && <div className="notice success">{saveStatus}</div>}
        {validation && (
          <div className={`notice ${validation.ok ? "success" : "error"}`}>
            {validation.ok ? "Valid feature config" : "Invalid feature config"} · {validation.estimated_layers} estimated layers
          </div>
        )}
        {validation?.errors.map((error) => (
          <div className="notice error" key={error}>{error}</div>
        ))}
        {validation?.warnings.map((warning) => (
          <div className="notice info" key={warning}>{warning}</div>
        ))}
        {features.length === 0 && <div className="notice info">Add at least one final feature before saving the run.</div>}
      </section>

      <section className="panel review-layers-panel">
        <div className="panel-head">
          <h2>Final Features</h2>
          <span className="field-hint">{expandedOutputs.length} expanded output{expandedOutputs.length === 1 ? "" : "s"}</span>
        </div>
        <div className="review-layer-list">
          {features.map((feature, index) => (
            <div className="review-layer-row derived-row" key={`${feature.name}-${index}`}>
              <span>
                <strong>{feature.title || feature.name}</strong>
                <small>{feature.name} · {feature.build_type} · {featureOutputCount(feature)} output{featureOutputCount(feature) === 1 ? "" : "s"}</small>
                {feature.outputs && feature.outputs.length > 1 && (
                  <span className="feature-chip-row compact-chips">
                    {feature.outputs.slice(0, 10).map((output) => (
                      <em key={output.name}>{output.suffix || output.name}</em>
                    ))}
                    {feature.outputs.length > 10 && <em>+{feature.outputs.length - 10}</em>}
                  </span>
                )}
              </span>
              <button className="ghost danger" onClick={() => removeFeature(index)}>Remove</button>
            </div>
          ))}
          {features.length === 0 && <div className="empty-state">No final features yet.</div>}
        </div>
      </section>

      <section className="panel yaml-panel">
        <h2>YAML</h2>
        <pre>{yamlText}</pre>
      </section>
    </main>
  );
}

function SourcesInfoPanel({ catalog }: { catalog: WorkbenchCatalog }) {
  const groupedSources = useMemo(() => {
    const groupMap = new Map<string, SourceCatalog[]>();
    for (const source of catalog.sources) {
      const key = source.provider ?? "other";
      groupMap.set(key, [...(groupMap.get(key) ?? []), source]);
    }
    const groupMeta = new Map((catalog.source_groups ?? []).map((group) => [group.id, group]));
    const groupOrder = new Map((catalog.source_groups ?? []).map((group, index) => [group.id, index]));
    return [...groupMap.entries()]
      .sort(([left], [right]) => {
        const leftOrder = groupOrder.get(left) ?? Number.MAX_SAFE_INTEGER;
        const rightOrder = groupOrder.get(right) ?? Number.MAX_SAFE_INTEGER;
        return leftOrder - rightOrder || left.localeCompare(right);
      })
      .map(([provider, sources]) => ({
        provider,
        meta: groupMeta.get(provider),
        sources: sources.sort((a, b) => sourceDisplayName(a).localeCompare(sourceDisplayName(b)))
      }));
  }, [catalog.source_groups, catalog.sources]);

  return (
    <main className="workspace sources-info-workspace">
      <section className="panel sources-overview-panel">
        <h2>Available Sources</h2>
        <p className="builder-copy">
          Browse source families first, then open each configured sub-source to inspect variables,
          temporal behaviour, units and links.
        </p>
      </section>
      <section className="source-info-tree">
        {groupedSources.map((group) => (
          <details key={group.provider} className="source-group source-family">
            <summary>
              <span>
                <strong>{group.meta?.title ?? humanizeId(group.provider)}</strong>
                <small>{group.meta?.summary ?? `${group.sources.length} configured sub-sources`}</small>
              </span>
              {group.meta?.official_url && (
                <a href={group.meta.official_url} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>
                  Official site
                </a>
              )}
            </summary>
            {group.meta?.long_description && <p className="source-group-description">{group.meta.long_description}</p>}
            {group.meta?.references && group.meta.references.length > 0 && (
              <div className="reference-links">
                {group.meta.references.map((reference) => (
                  <a key={reference.url} href={reference.url} target="_blank" rel="noreferrer">{reference.label}</a>
                ))}
              </div>
            )}
            <div className="source-stack nested-source-stack">
              {group.sources.map((source) => (
                <details key={source.id} className="source-card detailed subsource-card">
              <summary>
                <span>
                  <strong>{sourceDisplayName(source)}</strong>
                  <small>{sourceShortName(source)} · {providerDisplayName(source)}</small>
                </span>
                {sourceOfficialUrl(source) && <a href={sourceOfficialUrl(source)} target="_blank" rel="noreferrer">Official site</a>}
              </summary>
              <p className="source-group-description">{source.long_description ?? source.description ?? source.summary ?? "No extended description available."}</p>
              <div className="source-stack">
                {sourceDimensionEntries(source).length > 0 && (
                  <div className="mini-list">
                    {sourceDimensionEntries(source).map(([key, values]) => (
                      <span key={key}>{key}: {values.length} values</span>
                    ))}
                  </div>
                )}
                {source.temporal && (
                  <div className="notice info compact-notice">
                    Temporal model: {source.temporal.label ?? source.temporal.kind}
                  </div>
                )}
                {sourceVariables(source).map((variable) => (
                  <details key={variable.name} className="variable-card detailed">
                    <summary className="source-head">
                      <span className="source-title-row">
                        <strong>{variable.description ?? humanizeId(variable.name)}</strong>
                        <small>{variable.name} · {variable.kind}</small>
                      </span>
                      <em>{variable.unit ?? variable.geometry_type ?? variable.value_semantics ?? ""}</em>
                    </summary>
                    <div className="variable-detail-grid">
                      <span><strong>Type</strong>{variable.data_type ?? variable.value_semantics ?? "continuous"}</span>
                      <span><strong>Resampling</strong>{variable.resampling ?? "source default"}</span>
                      <span><strong>Native resolution</strong>{variable.native_resolution_m ? `${variable.native_resolution_m} m` : source.native_resolution ?? "source default"}</span>
                      {variable.valid_range && <span><strong>Valid range</strong>{variable.valid_range.join(" to ")}</span>}
                    </div>
                  </details>
                ))}
              </div>
            </details>
              ))}
            </div>
          </details>
        ))}
      </section>
    </main>
  );
}

interface ProjectPanelProps {
  catalog: WorkbenchCatalog | null;
  aois: AoiCatalog[];
  runName: string;
  setRunName: (value: string) => void;
  description: string;
  setDescription: (value: string) => void;
  projectConfig: string;
  setProjectConfig: (value: string) => void;
  targetCrs: string;
  setTargetCrs: (value: string) => void;
  aoiPath: string;
  setAoiPath: (value: string) => void;
  resolution: number;
  setResolution: (value: number) => void;
  stages: string[];
  setStages: (value: string[]) => void;
  datasetDir: string;
  setDatasetDir: (value: string) => void;
}

function ProjectPanel(props: ProjectPanelProps) {
  const aois = props.aois;
  const resolutions = props.catalog?.project.available_resolutions_m ?? [100];
  const supportedStages = props.catalog?.supported_stages ?? ["download", "clip", "build", "all"];

  return (
    <main className="workspace two-col">
      <section className="panel">
        <h2>Project Setup</h2>
        <div className="form-grid">
          <label>
            Run name
            <input value={props.runName} onChange={(event) => props.setRunName(event.target.value)} />
          </label>
          <label>
            Dataset directory
            <input value={props.datasetDir} onChange={(event) => props.setDatasetDir(event.target.value)} />
          </label>
          <label className="span-2">
            Description
            <input value={props.description} onChange={(event) => props.setDescription(event.target.value)} />
          </label>
          <label>
            Project config
            <input value={props.projectConfig} onChange={(event) => props.setProjectConfig(event.target.value)} />
          </label>
          <label>
            Output CRS
            <input value={props.targetCrs} onChange={(event) => props.setTargetCrs(event.target.value)} />
          </label>
          <label>
            AOI
            <select value={props.aoiPath} onChange={(event) => props.setAoiPath(event.target.value)}>
              {aois.map((aoi: AoiCatalog) => (
                <option key={aoi.path} value={aoi.path}>{aoi.name}</option>
              ))}
            </select>
          </label>
          <label>
            Resolution preset
            <select value={props.resolution} onChange={(event) => props.setResolution(Number(event.target.value))}>
              {!resolutions.includes(props.resolution) && (
                <option value={props.resolution}>{props.resolution} m</option>
              )}
              {resolutions.map((item) => (
                <option key={item} value={item}>{item} m</option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <section className="panel">
        <h2>Stages</h2>
        <div className="choice-list compact">
          {supportedStages.map((stage) => (
            <label key={stage} className="check-row">
              <input
                type="checkbox"
                checked={props.stages.includes(stage)}
                onChange={() => props.setStages(toggleStage(props.stages, stage))}
              />
              <span>{stage}</span>
            </label>
          ))}
        </div>
      </section>
    </main>
  );
}

function SourcePanel({
  catalog,
  selections,
  patchSelection,
  setActiveSourceId,
  setActiveTab
}: {
  catalog: WorkbenchCatalog;
  selections: Record<string, SourceSelection>;
  patchSelection: (sourceId: string, patch: Partial<SourceSelection>) => void;
  setActiveSourceId: (sourceId: string) => void;
  setActiveTab: (tab: string) => void;
}) {
  const [expandedInfo, setExpandedInfo] = useState<Record<string, boolean>>({});
  const groupedSources = useMemo(() => {
    const groupMap = new Map<string, SourceCatalog[]>();
    for (const source of catalog.sources) {
      const key = source.provider ?? "other";
      groupMap.set(key, [...(groupMap.get(key) ?? []), source]);
    }
    const groupMeta = new Map((catalog.source_groups ?? []).map((group) => [group.id, group]));
    const groupOrder = new Map((catalog.source_groups ?? []).map((group, index) => [group.id, index]));
    return [...groupMap.entries()].sort(([left], [right]) => {
      const leftOrder = groupOrder.get(left) ?? Number.MAX_SAFE_INTEGER;
      const rightOrder = groupOrder.get(right) ?? Number.MAX_SAFE_INTEGER;
      return leftOrder - rightOrder || left.localeCompare(right);
    }).map(([provider, sources]) => {
      const first = sources[0];
      const meta = groupMeta.get(provider);
      return {
        provider,
        title: meta?.title ?? (first ? providerDisplayName(first) : humanizeId(provider)),
        summary: meta?.summary,
        longDescription: meta?.long_description,
        officialUrl: meta?.official_url ?? first?.provider_url,
        sources: sources.sort((a, b) => sourceDisplayName(a).localeCompare(sourceDisplayName(b)))
      };
    });
  }, [catalog.source_groups, catalog.sources]);

  return (
    <main className="workspace">
      <section className="source-groups">
        {groupedSources.map((group) => (
          <details key={group.provider} className="source-group">
            <summary>
              <span>
                <strong>{group.title}</strong>
                <small>{group.summary ?? `${group.sources.length} configured sources`}</small>
              </span>
              {group.officialUrl && (
                <a href={group.officialUrl} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>
                  Official website
                </a>
              )}
            </summary>
            {group.longDescription && <p className="source-group-description">{group.longDescription}</p>}
            <div className="source-stack">
              {group.sources.map((source) => {
                const selection = selections[source.id];
                const sourceResolutionOptions = source.source_resolution_options?.length
                  ? source.source_resolution_options
                  : source.source_resolution
                    ? [source.source_resolution]
                    : [];
                const sourceResolution = selection?.sourceResolution ?? source.source_resolution ?? "";
                const officialUrl = sourceOfficialUrl(source);
                const isExpanded = Boolean(expandedInfo[source.id]);
                return (
                  <article key={source.id} className={`source-card detailed ${selection?.selected ? "selected" : ""}`}>
                    <div className="source-head">
                      <label className="switch-row source-title-row">
                        <input
                          type="checkbox"
                          checked={selection?.selected ?? false}
                          onChange={(event) => patchSelection(source.id, { selected: event.target.checked })}
                        />
                        <span>
                          <strong>{sourceDisplayName(source)}</strong>
                          <small>{source.id}</small>
                        </span>
                      </label>
                      <div className="source-actions">
                        <button
                          className="info-button"
                          aria-expanded={isExpanded}
                          aria-label={`More information about ${sourceDisplayName(source)}`}
                          onClick={() =>
                            setExpandedInfo((current) => ({
                              ...current,
                              [source.id]: !current[source.id]
                            }))
                          }
                        >
                          i
                        </button>
                        <button
                          className="ghost"
                          onClick={() => {
                            setActiveSourceId(source.id);
                            if (!selection?.selected) {
                              patchSelection(source.id, { selected: true });
                            }
                            setActiveTab("Variables");
                          }}
                        >
                          Edit
                        </button>
                      </div>
                    </div>
                    <div className="source-meta">
                      <MetaChip label="Source product" value={source.product_group ?? source.product} />
                      <MetaChip label="Source period" value={source.source_period ?? source.version ?? "current"} />
                      <MetaChip label="Source native resolution" value={source.native_resolution ?? source.source_resolution ?? "native"} />
                      <MetaChip label="Source CRS" value={source.source_crs} />
                    </div>
                    <p>{source.summary ?? source.description ?? source.layer_structure}</p>
                    {isExpanded && (
                      <div className="source-info-panel">
                        <p>{source.long_description ?? source.description ?? "No extended source notes are configured yet."}</p>
                        <dl>
                          <div>
                            <dt>Native structure</dt>
                            <dd>{source.layer_structure ?? source.file_format ?? "raster/vector source"}</dd>
                          </div>
                          <div>
                            <dt>Source data type</dt>
                            <dd>{source.data_type ?? "mixed/unspecified"}</dd>
                          </div>
                          <div>
                            <dt>Variables/layers</dt>
                            <dd>{sourceVariables(source).length}</dd>
                          </div>
                          {source.citation && (
                            <div>
                              <dt>Citation</dt>
                              <dd>{source.citation}</dd>
                            </div>
                          )}
                        </dl>
                        {officialUrl && (
                          <a href={officialUrl} target="_blank" rel="noreferrer">
                            Open official source page
                          </a>
                        )}
                      </div>
                    )}
                    {selection?.selected && (
                      <div className="source-controls">
                        <label className="switch-row subtle">
                          <input
                            type="checkbox"
                            checked={selection.stages.length === 0}
                            onChange={(event) =>
                              patchSelection(source.id, {
                                stages: event.target.checked ? [] : defaultStages(catalog.supported_stages)
                              })
                            }
                          />
                          <span>Use project stages</span>
                        </label>
                        {selection.stages.length > 0 && (
                          <div className="choice-list compact stage-grid">
                            {catalog.supported_stages.map((stage) => (
                              <label key={stage} className="check-row">
                                <input
                                  type="checkbox"
                                  checked={selection.stages.includes(stage)}
                                  onChange={() =>
                                    patchSelection(source.id, {
                                      stages: toggleStage(selection.stages, stage)
                                    })
                                  }
                                />
                                <span>{stage}</span>
                              </label>
                            ))}
                          </div>
                        )}
                        {sourceResolutionOptions.length > 0 && (
                          <label>
                            Source resolution
                            <select
                              value={sourceResolution}
                              onChange={(event) =>
                                patchSelection(source.id, {
                                  sourceResolution: event.target.value
                                })
                              }
                            >
                              {sourceResolutionOptions.map((item) => (
                                <option key={item} value={item}>{item}</option>
                              ))}
                            </select>
                          </label>
                        )}
                        <label className="switch-row subtle">
                          <input
                            type="checkbox"
                            checked={selection.keepRawAfterClip}
                            onChange={(event) =>
                              patchSelection(source.id, {
                                keepRawAfterClip: event.target.checked
                              })
                            }
                          />
                          <span>Keep raw after clip</span>
                        </label>
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          </details>
        ))}
      </section>
    </main>
  );
}

function NoSelectedSourcesPanel({ title }: { title: string }) {
  return (
    <main className="workspace">
      <section className="panel">
        <h2>{title}</h2>
        <div className="notice info">
          Select one or more sources before editing this section.
        </div>
      </section>
    </main>
  );
}

function SourceChooser({
  sources,
  activeSourceId,
  setActiveSourceId
}: {
  sources: SourceCatalog[];
  activeSourceId: string;
  setActiveSourceId: (value: string) => void;
}) {
  return (
    <label className="source-select">
      Source
      <select value={activeSourceId} onChange={(event) => setActiveSourceId(event.target.value)}>
        {sources.map((source) => (
          <option key={source.id} value={source.id}>{sourceShortName(source)}</option>
        ))}
      </select>
    </label>
  );
}

function VariablesPanel({
  catalog,
  sources,
  source,
  selection,
  activeSourceId,
  setActiveSourceId,
  patchSelectionMut
}: {
  catalog: WorkbenchCatalog;
  sources: SourceCatalog[];
  source: SourceCatalog;
  selection: SourceSelection;
  activeSourceId: string;
  setActiveSourceId: (value: string) => void;
  patchSelectionMut: (sourceId: string, edit: (selection: SourceSelection) => SourceSelection) => void;
}) {
  const variables = sourceVariables(source);
  const selectedVariableItems = variables.filter((variable) => selection.variables.includes(variable.name));
  const selectedFractionItems = selection.categoryFractions.map((fraction) => ({
    name: fraction.name,
    label: fraction.label ?? humanizeId(fraction.name),
    defaultMethod: "average",
    semantics: "fraction",
    sourceVariable: fraction.variable
  }));
  const dimensionEntries = sourceDimensionEntries(source);
  const fractionKey = (variableName: string, values: Array<string | number>) =>
    `${variableName}:${values.map(String).join(",")}`;
  const selectedFractionKeys = new Set(
    selection.categoryFractions.map((item) => fractionKey(item.variable, item.class_values))
  );

  function toggleCategoryFraction(variable: VariableCatalog, category: NonNullable<VariableCatalog["category_classes"]>[number]) {
    const values = categoryClassValues(category);
    if (values.length === 0) return;
    const key = fractionKey(variable.name, values);
    const selected = selectedFractionKeys.has(key);
    const label = category.label ?? category.name ?? values.join(", ");
    const name = `${variable.name}_fraction_${categoryClassToken(category)}`;

    patchSelectionMut(source.id, (current) => {
      const nextResampling = { ...current.resamplingByVariable };
      if (selected) {
        delete nextResampling[name];
      }
      return {
        ...current,
        resamplingByVariable: nextResampling,
        categoryFractions: selected
          ? current.categoryFractions.filter((item) => fractionKey(item.variable, item.class_values) !== key)
          : [
              ...current.categoryFractions,
              {
                variable: variable.name,
                name,
                class_values: values,
                label
              }
            ]
      };
    });
  }

  return (
    <main className="workspace two-col wide-left">
      <section className="panel">
        <div className="panel-head">
          <h2>Variables</h2>
          <SourceChooser sources={sources} activeSourceId={activeSourceId} setActiveSourceId={setActiveSourceId} />
        </div>
        <div className="choice-list">
          {variables.map((variable) => (
            <div key={variable.name} className="variable-card">
              <label className="check-row rich variable-main-row">
                <input
                  type="checkbox"
                  checked={selection.variables.includes(variable.name)}
                  onChange={() =>
                    patchSelectionMut(source.id, (current) => ({
                      ...current,
                      variables: toggleValue(current.variables, variable.name)
                    }))
                  }
                />
                <span>
                  <strong>{variable.description ?? humanizeId(variable.name)}</strong>
                  <small>{variable.name} · {variable.kind}</small>
                </span>
                <em>{variable.unit ?? variable.geometry_type ?? ""}</em>
              </label>
              <div className="variable-detail-grid">
                <span><strong>Semantics</strong>{variable.value_semantics ?? variable.data_type ?? "continuous"}</span>
                <span><strong>Default resampling</strong>{variable.resampling ?? "nearest"}</span>
                <span><strong>Native resolution</strong>{variable.native_resolution_m ? `${variable.native_resolution_m} m` : source.native_resolution ?? "source default"}</span>
                {variable.valid_range && (
                  <span><strong>Valid range</strong>{variable.valid_range.join(" to ")}</span>
                )}
                {variable.scale_factor !== undefined && variable.scale_factor !== 1 && (
                  <span><strong>Scale factor</strong>{variable.scale_factor}</span>
                )}
              </div>
              {(variable.temporal || variable.generated_from) && (
                <details className="inline-details">
                  <summary>Variable notes</summary>
                  {variable.generated_from && <p>Generated from: {variable.generated_from}</p>}
                  {variable.temporal && <pre>{JSON.stringify(variable.temporal, null, 2)}</pre>}
                </details>
              )}
              {(variable.category_classes ?? []).length > 0 && (
                <div className="category-fraction-panel">
                  <div className="category-fraction-head">
                    <strong>Category fractions</strong>
                    <small>Build 0-1 coverage rasters before target-grid resampling.</small>
                  </div>
                  <div className="choice-list compact token-grid">
                    {(variable.category_classes ?? []).map((category) => {
                      const values = categoryClassValues(category);
                      const key = fractionKey(variable.name, values);
                      return (
                        <label key={key} className="check-row">
                          <input
                            type="checkbox"
                            checked={selectedFractionKeys.has(key)}
                            onChange={() => toggleCategoryFraction(variable, category)}
                          />
                          <span>{category.label ?? category.name ?? values.join(", ")}</span>
                        </label>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Dimensions</h2>
        {dimensionEntries.length === 0 && (
          <div className="notice info">This source has no selectable dimensions.</div>
        )}
        {dimensionEntries.map(([key, values]) => {
          const selectedValues = selection.dimensions[key] ?? [];
          const allSelected = allValuesSelected(selectedValues, values);
          return (
          <div className="dimension-block" key={key}>
            <h3>{key}</h3>
            <div className="choice-list compact">
              <label className="check-row rich">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={() =>
                    patchSelectionMut(source.id, (current) => ({
                      ...current,
                      dimensions: {
                        ...current.dimensions,
                        [key]: allSelected ? [] : [...values]
                      }
                    }))
                  }
                />
                <span>
                  <strong>All</strong>
                  <small>{values.length} available values</small>
                </span>
              </label>
              {values.map((value) => (
                <label key={value} className="check-row">
                  <input
                    type="checkbox"
                    checked={selectedValues.includes(value)}
                    onChange={() =>
                      patchSelectionMut(source.id, (current) => ({
                        ...current,
                        dimensions: {
                          ...current.dimensions,
                          [key]: toggleValue(current.dimensions[key] ?? [], value)
                        }
                      }))
                    }
                  />
                  <span>{value}</span>
                </label>
              ))}
            </div>
          </div>
          );
        })}

        <div className="dimension-block">
          <h2>Resampling</h2>
          <div className="choice-list compact">
            {[
              ...selectedVariableItems.map((variable) => ({
                name: variable.name,
                label: variable.name,
                defaultMethod: variable.resampling ?? "nearest",
                semantics: variable.value_semantics ?? variable.data_type ?? "continuous",
                sourceVariable: undefined as string | undefined
              })),
              ...selectedFractionItems
            ].map((item) => {
              const currentMethod = selection.resamplingByVariable[item.name] ?? item.defaultMethod;
              return (
                <label key={item.name} className="resampling-row">
                  <span className="resampling-name">
                    {item.label}
                    {item.sourceVariable && <small>from {item.sourceVariable}</small>}
                  </span>
                  <select
                    value={currentMethod}
                    onChange={(event) =>
                      patchSelectionMut(source.id, (current) => {
                        const next = { ...current.resamplingByVariable };
                        if (event.target.value === item.defaultMethod) {
                          delete next[item.name];
                        } else {
                          next[item.name] = event.target.value;
                        }
                        return { ...current, resamplingByVariable: next };
                      })
                    }
                  >
                    {catalog.supported_resampling.map((method) => (
                      <option key={method} value={method}>
                        {method}{method === item.defaultMethod ? " (default)" : ""}
                      </option>
                    ))}
                  </select>
                  <small className="field-hint">
                    {item.semantics}
                  </small>
                </label>
              );
            })}
          </div>
        </div>
      </section>
    </main>
  );
}

function TemporalPanel({
  catalog,
  sources,
  source,
  selection,
  activeSourceId,
  setActiveSourceId,
  patchSelectionMut
}: {
  catalog: WorkbenchCatalog;
  sources: SourceCatalog[];
  source: SourceCatalog;
  selection: SourceSelection;
  activeSourceId: string;
  setActiveSourceId: (value: string) => void;
  patchSelectionMut: (sourceId: string, edit: (selection: SourceSelection) => SourceSelection) => void;
}) {
  const capability = source.temporal;
  const temporal = selection.temporal;
  const isTimeSeries = capability?.kind === "year_month_series";
  const isYearlyCollection = capability?.kind === "yearly_static_collection";
  const isPostprocess = capability?.kind === "temporal_postprocess";
  const postprocessMetricOptions = capability?.postprocess_metrics ?? [
    "mean",
    "std",
    "min",
    "max",
    "count_threshold",
    "valid_observation_count"
  ];
  const defaultMetric = isPostprocess ? postprocessMetricOptions[0] ?? "mean" : "mean";
  const defaultForm = isYearlyCollection
    ? "year_range_metric"
    : isPostprocess
      ? "explicit_month_list_metric"
      : isTimeSeries
      ? "year_then_across_years"
      : "month_range_metric";
  const [custom, setCustom] = useState<CustomAggregation>({
    name: isYearlyCollection ? "custom_year_mean" : "custom_mean",
    form: defaultForm,
    metric: defaultMetric,
    months: isYearlyCollection ? undefined : [1, 12],
    years: temporal.years,
    within_year_metric: "sum",
    across_year_metric: "mean",
    threshold: 50,
    comparison: ">=",
    variables: selection.variables.slice(0, 1)
  });

  useEffect(() => {
    setCustom({
      name: isYearlyCollection ? "custom_year_mean" : isTimeSeries ? "custom_period" : isPostprocess ? "custom_snow_metric" : "custom_mean",
      form: isYearlyCollection ? "year_range_metric" : isTimeSeries ? "year_then_across_years" : isPostprocess ? "explicit_month_list_metric" : "month_range_metric",
      metric: isPostprocess ? postprocessMetricOptions[0] ?? "mean" : "mean",
      months: isYearlyCollection ? undefined : temporal.months,
      years: temporal.years,
      within_year_metric: "sum",
      across_year_metric: "mean",
      output_metric_name: isTimeSeries ? "mean_period_sum" : undefined,
      threshold: 50,
      comparison: ">=",
      variables: selection.variables.slice(0, 1)
    });
  }, [source.id]);

  const selectedPostprocessVariable = custom.variables.find((variable) => selection.variables.includes(variable)) ?? selection.variables[0] ?? "";
  const selectedCustomVariables = isPostprocess
    ? (selectedPostprocessVariable ? [selectedPostprocessVariable] : [])
    : custom.variables.filter((variable) => selection.variables.includes(variable));
  const canAddCustom = custom.name.trim().length > 0 && selectedCustomVariables.length > 0;
  const supportsAggregate = capability?.output_modes.includes("aggregate") ?? false;
  const supportsRaw = capability?.output_modes.includes("raw_slices") ?? false;
  const supportsSupplied = capability?.output_modes.includes("supplied_layers") ?? false;
  const supportsPostprocess = capability?.output_modes.includes("postprocess_aggregate") ?? false;
  const availableYearStart = capability?.available_years?.[0] ?? capability?.default_years?.[0];
  const availableYearEnd = capability?.available_years?.[1] ?? capability?.default_years?.[1];
  const yearlyAggregationYears = isYearlyCollection ? capability?.temporal_layers?.years ?? [] : [];
  const useDiscreteYearSelect = yearlyAggregationYears.length > 0;
  const firstDiscreteYear = yearlyAggregationYears[0];
  const lastDiscreteYear = yearlyAggregationYears[yearlyAggregationYears.length - 1];
  const customStartYear = custom.years?.[0] ?? firstDiscreteYear ?? "";
  const customEndYear = custom.years?.[1] ?? lastDiscreteYear ?? "";
  const dateMin = availableYearStart ? `${availableYearStart}-01-01` : undefined;
  const dateMax = availableYearEnd ? `${availableYearEnd}-12-31` : undefined;
  const clampMonth = (value: number) => clamp(Number.isFinite(value) ? value : 1, 1, 12);
  const clampYear = (value: number) => {
    let next = Number.isFinite(value) ? value : availableYearStart ?? new Date().getFullYear();
    if (availableYearStart !== undefined) next = Math.max(availableYearStart, next);
    if (availableYearEnd !== undefined) next = Math.min(availableYearEnd, next);
    return next;
  };
  const setCustomYearStart = (value: number) => {
    const next = useDiscreteYearSelect ? value : clampYear(value);
    const currentEnd = typeof custom.years?.[1] === "number" ? custom.years[1] : next;
    setCustom({ ...custom, years: [next, currentEnd < next ? next : currentEnd] });
  };
  const setCustomYearEnd = (value: number) => {
    const next = useDiscreteYearSelect ? value : clampYear(value);
    const currentStart = typeof custom.years?.[0] === "number" ? custom.years[0] : next;
    setCustom({ ...custom, years: [currentStart > next ? next : currentStart, next] });
  };

  function patchTemporal(patch: Partial<TemporalSelection>) {
    patchSelectionMut(source.id, (current) => ({
      ...current,
      temporal: {
        ...current.temporal,
        ...patch
      }
    }));
  }

  function patchTemporalLayers(patch: Partial<TemporalSelection["layers"]>) {
    patchSelectionMut(source.id, (current) => ({
      ...current,
      temporal: {
        ...current.temporal,
        layers: {
          ...current.temporal.layers,
          ...patch
        }
      }
    }));
  }

  const aggregationPresetPanel = supportsAggregate ? (
    <section className="panel preset-panel">
      <h2>Presets</h2>
      <div className="choice-list compact">
        {(source.aggregations ?? []).map((aggregation) => {
          const name = String(aggregation.name);
          return (
            <label key={name} className="check-row rich preset-row">
              <input
                type="checkbox"
                checked={temporal.aggregationUse.includes(name)}
                onChange={() =>
                  patchSelectionMut(source.id, (current) => ({
                    ...current,
                    temporal: {
                      ...current.temporal,
                      aggregationUse: toggleValue(current.temporal.aggregationUse, name)
                    }
                  }))
                }
              />
              <span>
                <strong>{name}</strong>
                <small>{String(aggregation.metric ?? aggregation.output_metric_name ?? "two-step")}</small>
              </span>
              <em>
                {Array.isArray(aggregation.years) ? `y${aggregation.years.join("-")} ` : ""}
                {Array.isArray(aggregation.months) ? `m${aggregation.months.join("-")}` : ""}
              </em>
            </label>
          );
        })}
      </div>
      {(source.aggregations ?? []).length === 0 && (
        <div className="notice info">
          This source has no predefined build-time aggregations.
        </div>
      )}
      {temporal.aggregationUse.length > 0 && (
        <div className="aggregation-list compact-summary">
          {temporal.aggregationUse.map((name) => (
            <div className="aggregation-chip preset-chip" key={name}>
              <span>{name}</span>
              <button
                className="ghost danger"
                onClick={() =>
                  patchSelectionMut(source.id, (current) => ({
                    ...current,
                    temporal: {
                      ...current.temporal,
                      aggregationUse: current.temporal.aggregationUse.filter((item) => item !== name)
                    }
                  }))
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  ) : null;

  return (
    <main className="workspace two-col wide-left">
      <section className="panel custom-aggregation-panel">
        <div className="panel-head">
          <h2>Temporal Output</h2>
          <SourceChooser sources={sources} activeSourceId={activeSourceId} setActiveSourceId={setActiveSourceId} />
        </div>

        <div className="temporal-kind">
          <strong>{capability?.label ?? source.layer_structure ?? "Source temporal model"}</strong>
          <span>{capability?.kind ?? "unknown"}</span>
        </div>
        {capability?.note && (
          <div className="notice info compact-notice">
            {capability.note}
          </div>
        )}

        <div className="form-grid temporal-mode-grid">
          <label>
            Output mode
            <select value={temporal.outputMode} onChange={(event) => patchTemporal({ outputMode: event.target.value })}>
              {(capability?.output_modes ?? ["static"]).map((mode) => (
                <option key={mode} value={mode}>{mode}</option>
              ))}
            </select>
          </label>
        </div>

        {temporal.outputMode === "static" && (
          <div className="notice info">
            This source is spatial only for the current pipeline. Variables are written directly without temporal aggregation.
          </div>
        )}

        {temporal.outputMode === "raw_slices" && supportsRaw && (
          <div className="form-grid custom-aggregation-grid">
            {isTimeSeries && (
              <>
                <label>
                  Start year
                  <input type="number" value={temporal.years?.[0] ?? ""} onChange={(event) => patchTemporal({ years: [Number(event.target.value), temporal.years?.[1] ?? Number(event.target.value)] })} />
                </label>
                <label>
                  End year
                  <input type="number" value={temporal.years?.[1] ?? ""} onChange={(event) => patchTemporal({ years: [temporal.years?.[0] ?? Number(event.target.value), Number(event.target.value)] })} />
                </label>
              </>
            )}
            <label>
              Start month
              <input type="number" min={1} max={12} value={temporal.months[0]} onChange={(event) => patchTemporal({ months: [Number(event.target.value), temporal.months[1]] })} />
            </label>
            <label>
              End month
              <input type="number" min={1} max={12} value={temporal.months[1]} onChange={(event) => patchTemporal({ months: [temporal.months[0], Number(event.target.value)] })} />
            </label>
          </div>
        )}

        {temporal.outputMode === "supplied_layers" && supportsSupplied && (
          <div className="temporal-layer-editor">
            <div className="notice info compact-notice">
              Select the supplied temporal layers you want to build. Leaving every option empty builds no temporal layer for this source.
            </div>
            <div className="choice-list compact">
              {capability?.temporal_layers?.annual && (
              <label className="check-row rich">
                <input
                  type="checkbox"
                  checked={temporal.layers.annual}
                  onChange={() => patchTemporalLayers({ annual: !temporal.layers.annual })}
                />
                <span>
                  <strong>Annual summary layers</strong>
                  <small>One raster per variable summarizing the whole year.</small>
                </span>
              </label>
              )}
              {capability?.temporal_layers?.annual_index && (
              <label className="check-row rich">
                <input
                  type="checkbox"
                  checked={temporal.layers.annual_index}
                  onChange={() => patchTemporalLayers({ annual_index: !temporal.layers.annual_index })}
                />
                <span>
                  <strong>Annual index layers</strong>
                  <small>Year-level index rasters supplied directly by the source.</small>
                </span>
              </label>
              )}
            </div>
            {(capability?.temporal_layers?.years ?? []).length > 0 && (
              <>
                <h3>Years</h3>
                <div className="choice-list compact token-grid">
                  <label className="check-row rich">
                    <input
                      type="checkbox"
                      checked={allValuesSelected(temporal.layers.years, capability?.temporal_layers?.years ?? [])}
                      onChange={() => {
                        const years = capability?.temporal_layers?.years ?? [];
                        patchTemporalLayers({
                          years: allValuesSelected(temporal.layers.years, years) ? [] : [...years]
                        });
                      }}
                    />
                    <span>
                      <strong>All years</strong>
                      <small>{(capability?.temporal_layers?.years ?? []).length} available years</small>
                    </span>
                  </label>
                  {(capability?.temporal_layers?.years ?? []).map((year) => (
                    <label key={year} className="check-row">
                      <input
                        type="checkbox"
                        checked={temporal.layers.years.includes(year)}
                        onChange={() =>
                          patchTemporalLayers({
                            years: temporal.layers.years.includes(year)
                              ? temporal.layers.years.filter((item) => item !== year)
                              : [...temporal.layers.years, year].sort((left, right) => left - right)
                          })
                        }
                      />
                      <span>{year}</span>
                    </label>
                  ))}
                </div>
              </>
            )}
            {(capability?.temporal_layers?.months ?? []).length > 0 && (
              <>
                <h3>Months</h3>
                <div className="choice-list compact token-grid">
                  <label className="check-row rich">
                    <input
                      type="checkbox"
                      checked={allValuesSelected(temporal.layers.months, capability?.temporal_layers?.months ?? [])}
                      onChange={() => {
                        const months = capability?.temporal_layers?.months ?? [];
                        patchTemporalLayers({
                          months: allValuesSelected(temporal.layers.months, months) ? [] : [...months]
                        });
                      }}
                    />
                    <span>
                      <strong>All months</strong>
                      <small>{(capability?.temporal_layers?.months ?? []).length} available months</small>
                    </span>
                  </label>
                  {(capability?.temporal_layers?.months ?? []).map((month) => (
                    <label key={month} className="check-row">
                      <input
                        type="checkbox"
                        checked={temporal.layers.months.includes(month)}
                        onChange={() => patchTemporalLayers({ months: toggleValue(temporal.layers.months, month) })}
                      />
                      <span>{month}</span>
                    </label>
                  ))}
                </div>
              </>
            )}
            {(capability?.temporal_layers?.seasons ?? []).length > 0 && (
              <>
                <h3>Seasons</h3>
                <div className="choice-list compact token-grid">
                  <label className="check-row rich">
                    <input
                      type="checkbox"
                      checked={allValuesSelected(temporal.layers.seasons, capability?.temporal_layers?.seasons ?? [])}
                      onChange={() => {
                        const seasons = capability?.temporal_layers?.seasons ?? [];
                        patchTemporalLayers({
                          seasons: allValuesSelected(temporal.layers.seasons, seasons) ? [] : [...seasons]
                        });
                      }}
                    />
                    <span>
                      <strong>All seasons</strong>
                      <small>{(capability?.temporal_layers?.seasons ?? []).length} available seasons</small>
                    </span>
                  </label>
                  {(capability?.temporal_layers?.seasons ?? []).map((season) => (
                    <label key={season} className="check-row">
                      <input
                        type="checkbox"
                        checked={temporal.layers.seasons.includes(season)}
                        onChange={() => patchTemporalLayers({ seasons: toggleValue(temporal.layers.seasons, season) })}
                      />
                      <span>{season}</span>
                    </label>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {temporal.outputMode === "postprocess_aggregate" && supportsPostprocess && (
          <>
            <div className="notice info">
              This source creates temporal products during download/postprocess. Select presets or define custom outputs here.
            </div>
            <h3>Presets</h3>
            <div className="choice-list compact">
              {(capability?.postprocess_outputs ?? []).map((item) => (
                <label key={String(item.name)} className="check-row rich preset-row">
                  <input
                    type="checkbox"
                    checked={temporal.aggregationUse.includes(String(item.name))}
                    onChange={() =>
                      patchSelectionMut(source.id, (current) => ({
                        ...current,
                        temporal: {
                          ...current.temporal,
                          aggregationUse: toggleValue(current.temporal.aggregationUse, String(item.name))
                        }
                      }))
                    }
                  />
                  <span>
                    <strong>{String(item.name)}</strong>
                    <small>
                      {String(item.method ?? "")}
                      {Array.isArray(item.months) ? ` · months ${item.months.join(", ")}` : ""}
                      {Array.isArray(item.years) ? ` · years ${item.years.join("-")}` : ""}
                    </small>
                  </span>
                </label>
              ))}
            </div>
            {(capability?.postprocess_outputs ?? []).length === 0 && (
              <div className="notice info compact-notice">
                This source has no predefined postprocess presets.
              </div>
            )}

            <h3>Custom postprocess aggregation</h3>
            <div className="form-grid custom-aggregation-grid postprocess-aggregation-grid">
              <label className="postprocess-name-field">
                Name
                <input value={custom.name} onChange={(event) => setCustom({ ...custom, name: event.target.value })} />
              </label>
              <label className="postprocess-metric-field">
                Metric
                <select value={custom.metric} onChange={(event) => setCustom({ ...custom, metric: event.target.value })}>
                  {postprocessMetricOptions.map((metric) => (
                    <option key={metric} value={metric}>{metric}</option>
                  ))}
                </select>
              </label>
              {capability?.available_years && (
                <>
                  <label className="postprocess-year-start-field">
                    Start year
                    <input
                      type="number"
                      min={availableYearStart}
                      max={availableYearEnd}
                      value={custom.years?.[0] ?? capability.available_years[0]}
                      onChange={(event) => {
                        const next = clampYear(Number(event.target.value));
                        setCustom({ ...custom, years: [next, custom.years?.[1] ?? next] });
                      }}
                    />
                  </label>
                  <label className="postprocess-year-end-field">
                    End year
                    <input
                      type="number"
                      min={availableYearStart}
                      max={availableYearEnd}
                      value={custom.years?.[1] ?? capability.available_years[1]}
                      onChange={(event) => {
                        const next = clampYear(Number(event.target.value));
                        setCustom({ ...custom, years: [custom.years?.[0] ?? next, next] });
                      }}
                    />
                  </label>
                </>
              )}
              <label className="postprocess-month-start-field">
                Start month
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={custom.months?.[0] ?? 1}
                  onChange={(event) =>
                    setCustom({
                      ...custom,
                      months: [clampMonth(Number(event.target.value)), custom.months?.[1] ?? 12]
                    })
                  }
                />
              </label>
              <label className="postprocess-month-end-field">
                End month
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={custom.months?.[1] ?? 12}
                  onChange={(event) =>
                    setCustom({
                      ...custom,
                      months: [custom.months?.[0] ?? 1, clampMonth(Number(event.target.value))]
                    })
                  }
                />
                <small className="field-hint">Use 12 to 3 for a wrapped winter range.</small>
              </label>
              <label className="postprocess-date-start-field">
                Exact start date
                <input
                  type="date"
                  min={dateMin}
                  max={dateMax}
                  value={custom.start_date ?? ""}
                  onChange={(event) => setCustom({ ...custom, start_date: event.target.value || undefined })}
                />
                <small className="field-hint">Optional; useful for fortnight windows.</small>
              </label>
              <label className="postprocess-date-end-field">
                Exact end date
                <input
                  type="date"
                  min={dateMin}
                  max={dateMax}
                  value={custom.end_date ?? ""}
                  onChange={(event) => setCustom({ ...custom, end_date: event.target.value || undefined })}
                />
              </label>
              {custom.metric === "count_threshold" && (
                <>
                  <label>
                    Threshold
                    <input type="number" value={custom.threshold ?? 50} onChange={(event) => setCustom({ ...custom, threshold: Number(event.target.value) })} />
                  </label>
                  <label>
                    Comparison
                    <select value={custom.comparison ?? ">="} onChange={(event) => setCustom({ ...custom, comparison: event.target.value })}>
                      <option value=">=">&gt;=</option>
                      <option value=">">&gt;</option>
                      <option value="<=">&lt;=</option>
                      <option value="<">&lt;</option>
                      <option value="==">==</option>
                    </select>
                  </label>
                </>
              )}
            </div>
            <h3>Input variable</h3>
            {selection.variables.length > 1 ? (
              <div className="form-grid compact-grid">
                <label>
                  Source variable
                  <select
                    value={selectedPostprocessVariable}
                    onChange={(event) => setCustom({ ...custom, variables: event.target.value ? [event.target.value] : [] })}
                  >
                    {selection.variables.map((variable) => (
                      <option key={variable} value={variable}>{variable}</option>
                    ))}
                  </select>
                </label>
              </div>
            ) : selection.variables.length === 1 ? (
              <div className="notice info compact-notice">
                Input variable: {selection.variables[0]}
              </div>
            ) : (
              <div className="notice error compact-notice">
                Select at least one source variable before defining a postprocess aggregation.
              </div>
            )}
            <div className="button-row custom-actions">
              <button
                className="primary"
                disabled={!canAddCustom}
                onClick={() => {
                  if (!canAddCustom) return;
                  const nextCustom = {
                    ...custom,
                    name: custom.name.trim(),
                    variables: selectedCustomVariables
                  };
                  patchSelectionMut(source.id, (current) => ({
                    ...current,
                    temporal: {
                      ...current.temporal,
                      customAggregations: [...current.temporal.customAggregations, nextCustom]
                    }
                  }));
                  setCustom({
                    ...custom,
                    name: `${custom.name.trim() || "custom_snow_metric"}_copy`
                  });
                }}
              >
                Add postprocess aggregation
              </button>
            </div>
            <div className="aggregation-list">
              {temporal.aggregationUse.map((name) => (
                <div className="aggregation-chip preset-chip" key={name}>
                  <span>{name}</span>
                  <button
                    className="ghost danger"
                    onClick={() =>
                      patchSelectionMut(source.id, (current) => ({
                        ...current,
                        temporal: {
                          ...current.temporal,
                          aggregationUse: current.temporal.aggregationUse.filter((item) => item !== name)
                        }
                      }))
                    }
                  >
                    Remove
                  </button>
                </div>
              ))}
              {temporal.customAggregations.map((item, index) => (
                <div className="aggregation-chip" key={`${item.name}-${index}`}>
                  <span>
                    <strong>{item.name}</strong>
                    <small>
                      {item.metric}
                      {item.threshold !== undefined && item.metric === "count_threshold" ? ` ${item.comparison ?? ">="} ${item.threshold}` : ""}
                      {item.years ? ` · years ${item.years.join("-")}` : ""}
                      {item.months ? ` · months ${item.months.join("-")}` : ""} · {item.variables.join(", ")}
                    </small>
                  </span>
                  <button
                    className="ghost danger"
                    onClick={() =>
                      patchSelectionMut(source.id, (current) => ({
                        ...current,
                        temporal: {
                          ...current.temporal,
                          customAggregations: current.temporal.customAggregations.filter((_, itemIndex) => itemIndex !== index)
                        }
                      }))
                    }
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </>
        )}

        {temporal.outputMode === "aggregate" && supportsAggregate && (
          <>
            <h3>Custom aggregation</h3>
            <div className="form-grid custom-aggregation-grid">
              <label>
                Name
                <input value={custom.name} onChange={(event) => setCustom({ ...custom, name: event.target.value })} />
              </label>
              {isTimeSeries && (
                <label>
                  Form
                  <select value={custom.form} onChange={(event) => setCustom({ ...custom, form: event.target.value })}>
                    <option value="year_range_month_range_metric">Direct across year-months</option>
                    <option value="year_then_across_years">Yearly then across years</option>
                  </select>
                </label>
              )}
              {(isTimeSeries || isYearlyCollection) && (
                <>
                  <label>
                    Start year
                    {useDiscreteYearSelect ? (
                      <select
                        value={customStartYear}
                        onChange={(event) => setCustomYearStart(Number(event.target.value))}
                      >
                        {yearlyAggregationYears.map((year) => (
                          <option key={year} value={year}>{year}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        min={availableYearStart}
                        max={availableYearEnd}
                        value={custom.years?.[0] ?? ""}
                        onChange={(event) => setCustomYearStart(Number(event.target.value))}
                      />
                    )}
                  </label>
                  <label>
                    End year
                    {useDiscreteYearSelect ? (
                      <select
                        value={customEndYear}
                        onChange={(event) => setCustomYearEnd(Number(event.target.value))}
                      >
                        {yearlyAggregationYears.map((year) => (
                          <option key={year} value={year}>{year}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        min={availableYearStart}
                        max={availableYearEnd}
                        value={custom.years?.[1] ?? ""}
                        onChange={(event) => setCustomYearEnd(Number(event.target.value))}
                      />
                    )}
                  </label>
                </>
              )}
              {(!isTimeSeries || custom.form === "year_range_month_range_metric") && (
                <label>
                  Metric
                  <select value={custom.metric} onChange={(event) => setCustom({ ...custom, metric: event.target.value })}>
                    {catalog.supported_metrics.map((metric) => (
                      <option key={metric} value={metric}>{metric}</option>
                    ))}
                  </select>
                </label>
              )}
              {isTimeSeries && custom.form === "year_then_across_years" && (
                <>
                  <label>
                    Within-year metric
                    <select value={custom.within_year_metric ?? "sum"} onChange={(event) => setCustom({ ...custom, within_year_metric: event.target.value })}>
                      {catalog.supported_metrics.map((metric) => (
                        <option key={metric} value={metric}>{metric}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Across-years metric
                    <select value={custom.across_year_metric ?? "mean"} onChange={(event) => setCustom({ ...custom, across_year_metric: event.target.value })}>
                      {catalog.supported_metrics.map((metric) => (
                        <option key={metric} value={metric}>{metric}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Output metric name
                    <input value={custom.output_metric_name ?? ""} onChange={(event) => setCustom({ ...custom, output_metric_name: event.target.value || undefined })} />
                  </label>
                </>
              )}
              {!isYearlyCollection && (
                <>
                  <label>
                    Start month
                    <input
                      type="number"
                      min={1}
                      max={12}
                      value={custom.months?.[0] ?? 1}
                      onChange={(event) =>
                        setCustom({
                          ...custom,
                          months: [clampMonth(Number(event.target.value)), custom.months?.[1] ?? 12]
                        })
                      }
                    />
                  </label>
                  <label>
                    End month
                    <input
                      type="number"
                      min={1}
                      max={12}
                      value={custom.months?.[1] ?? 12}
                      onChange={(event) =>
                        setCustom({
                          ...custom,
                          months: [custom.months?.[0] ?? 1, clampMonth(Number(event.target.value))]
                        })
                      }
                    />
                  </label>
                </>
              )}
            </div>
            <h3>Variables</h3>
            <div className="choice-list compact custom-vars">
              {selection.variables.map((variable) => (
                <label key={variable} className="check-row">
                  <input
                    type="checkbox"
                    checked={custom.variables.includes(variable)}
                    onChange={() => setCustom({ ...custom, variables: toggleValue(custom.variables, variable) })}
                  />
                  <span>{variable}</span>
                </label>
              ))}
            </div>
            <div className="button-row custom-actions">
              <button
                className="primary"
                disabled={!canAddCustom}
                onClick={() => {
                  if (!canAddCustom) return;
                  const nextCustom = {
                    ...custom,
                    name: custom.name.trim(),
                    variables: selectedCustomVariables
                  };
                  patchSelectionMut(source.id, (current) => ({
                    ...current,
                    temporal: {
                      ...current.temporal,
                      customAggregations: [...current.temporal.customAggregations, nextCustom]
                    }
                  }));
                  setCustom({
                    ...custom,
                    name: `${custom.name.trim() || "custom_mean"}_copy`
                  });
                }}
              >
                Add aggregation
              </button>
            </div>
            <div className="aggregation-list">
              {temporal.customAggregations.map((item, index) => (
                <div className="aggregation-chip" key={`${item.name}-${index}`}>
                  <span>
                    <strong>{item.name}</strong>
                    <small>
                      {item.form === "year_then_across_years"
                        ? `${item.within_year_metric ?? "sum"} then ${item.across_year_metric ?? "mean"}`
                        : item.metric}
                      {item.years ? ` · years ${item.years.join("-")}` : ""}
                      {item.months ? ` · months ${item.months.join("-")}` : ""} · {item.variables.join(", ")}
                    </small>
                  </span>
                <button
                  className="ghost danger"
                  onClick={() =>
                    patchSelectionMut(source.id, (current) => ({
                      ...current,
                      temporal: {
                        ...current.temporal,
                        customAggregations: current.temporal.customAggregations.filter((_, itemIndex) => itemIndex !== index)
                      }
                    }))
                  }
                >
                  Remove
                </button>
              </div>
              ))}
            </div>
          </>
        )}
      </section>

      {aggregationPresetPanel ?? (
        <section className="panel preset-panel">
          <h2>Temporal Notes</h2>
          <div className="notice info">
            This source does not use build-time aggregation presets.
          </div>
        </section>
      )}
    </main>
  );
}

interface PlannedLayer {
  id: string;
  label: string;
  sourceTitle: string;
  query: DerivedInputQuery;
  variable: string;
  baseVariable?: string;
  unit?: string | null;
  valueSemantics?: string;
}

function queryMatchesLayer(layer: PlannedLayer, query: DerivedInputQuery) {
  return Object.entries(query).every(([key, value]) => {
    if (value === undefined || value === null) return true;
    return layer.query[key as keyof DerivedInputQuery] === value;
  });
}

function derivedFeatureDependsOnLayer(feature: DerivedFeatureConfig, layer: PlannedLayer) {
  return Object.values(feature.inputs ?? {}).some((query) => queryMatchesLayer(layer, query));
}

function buildPlannedLayers(
  catalog: WorkbenchCatalog,
  selectedSources: SourceSelection[]
): PlannedLayer[] {
  const layers: PlannedLayer[] = [];
  const dimensionValues = (selection: SourceSelection, key: string) => {
    if (Object.prototype.hasOwnProperty.call(selection.dimensions, key)) {
      return selection.dimensions[key] ?? [];
    }
    return [undefined];
  };

  for (const selection of selectedSources) {
    const source = catalog.sources.find((item) => item.id === selection.id);
    if (!source) continue;

    const variables = sourceVariables(source).filter((item) => selection.variables.includes(item.name));
    const aggregations = [
      ...selection.temporal.aggregationUse,
      ...selection.temporal.customAggregations.map((item) => item.name)
    ];
    const temporalAggregations = selection.temporal.outputMode === "aggregate" && aggregations.length > 0
      ? aggregations
      : [undefined];
    const gcms = dimensionValues(selection, "gcms");
    const ssps = dimensionValues(selection, "ssps");
    const periods = dimensionValues(selection, "periods");

    for (const variable of variables) {
      const variablePattern = typeof variable.temporal?.variable_pattern === "string"
        ? variable.temporal.variable_pattern
        : undefined;
      const yearlySuppliedYears = source.temporal?.kind === "yearly_static_collection" &&
        selection.temporal.outputMode === "supplied_layers"
        ? selection.temporal.layers.years
        : [];
      const patternContexts = dimensionPatternContexts(source, selection);

      if (variablePattern && yearlySuppliedYears.length > 0) {
        for (const year of yearlySuppliedYears) {
          for (const context of patternContexts) {
            const expandedVariable = applyVariablePattern(variablePattern, { ...context, year });
            const query: DerivedInputQuery = {
              source_id: source.id,
              variable: expandedVariable
            };
            layers.push({
              id: JSON.stringify(query),
              label: [sourceShortName(source), variable.name, ...Object.values(context), year].join(" · "),
              sourceTitle: sourceDisplayName(source),
              query,
              variable: expandedVariable,
              baseVariable: variable.name,
              unit: variable.unit,
              valueSemantics: variable.value_semantics ?? variable.data_type
            });
          }
        }
        continue;
      }

      for (const aggregation of temporalAggregations) {
        for (const gcm of gcms) {
          for (const ssp of ssps) {
            for (const period of periods) {
              const query: DerivedInputQuery = {
                source_id: source.id,
                variable: variable.name,
                aggregation_name: aggregation,
                gcm,
                ssp,
                period
              };
              const labelBits = [
                sourceShortName(source),
                variable.name,
                aggregation,
                gcm,
                ssp,
                period
              ].filter(Boolean);
              layers.push({
                id: JSON.stringify(query),
                label: labelBits.join(" · "),
                sourceTitle: sourceDisplayName(source),
                query,
                variable: variable.name,
                unit: variable.unit,
                valueSemantics: variable.value_semantics ?? variable.data_type
              });
            }
          }
        }
      }
    }

    for (const fraction of selection.categoryFractions) {
      const query: DerivedInputQuery = {
        source_id: source.id,
        variable: fraction.name
      };
      layers.push({
        id: JSON.stringify(query),
        label: [sourceShortName(source), fraction.name].join(" · "),
        sourceTitle: sourceDisplayName(source),
        query,
        variable: fraction.name,
        baseVariable: fraction.variable,
        unit: "fraction",
        valueSemantics: "fraction"
      });
    }
  }

  return layers;
}

function sanitizeDerivedName(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/^_+|_+$/g, "")
    || "derived_feature";
}

function isDemLayer(layer?: PlannedLayer) {
  if (!layer) return false;
  const tokens = `${layer.variable} ${layer.label}`.toLowerCase();
  return /\b(dem|elev|elevation|altitude|height|glo-30|glo30)\b/.test(tokens);
}

function layerTokens(layer?: PlannedLayer) {
  if (!layer) return "";
  return `${layer.variable} ${layer.label} ${layer.valueSemantics ?? ""} ${layer.unit ?? ""}`.toLowerCase();
}

function layerVariableMatches(layer: PlannedLayer, names: string[]) {
  const variable = layer.variable.toLowerCase();
  const label = layer.label.toLowerCase();
  return names.some((name) => {
    const normalized = name.toLowerCase();
    const tokenPattern = new RegExp(`(^|[^a-z0-9])${normalized.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^a-z0-9]|$)`);
    return variable === normalized || tokenPattern.test(variable) || tokenPattern.test(label);
  });
}

function isCategoricalLayer(layer?: PlannedLayer) {
  const tokens = layerTokens(layer);
  return /\b(categorical|ordinal|binary|mask|class|classes|landcover|land_cover|corine|forest|road|roads|track|building|settlement)\b/.test(tokens);
}

function isBinaryLayer(layer?: PlannedLayer) {
  const tokens = layerTokens(layer);
  return /\b(binary|mask|presence|absence|0\/1)\b/.test(tokens);
}

function isNumericLayer(layer?: PlannedLayer) {
  if (!layer) return false;
  return !isCategoricalLayer(layer) || /\b(count|percentage|fraction|ratio|intensive|extensive|depth|temperature|precipitation|pet|biomass|density|distance|slope|aspect)\b/.test(layerTokens(layer));
}

function sameLayerContext(left?: PlannedLayer, right?: PlannedLayer) {
  if (!left || !right) return true;
  return (
    left.query.source_id === right.query.source_id &&
    left.query.aggregation_name === right.query.aggregation_name &&
    left.query.gcm === right.query.gcm &&
    left.query.ssp === right.query.ssp &&
    left.query.period === right.query.period
  );
}

const recipeHelp: Record<string, string> = {
  thermal_range: "Difference between maximum and minimum temperature. Useful as a simple temperature variability layer.",
  water_balance: "Precipitation minus potential evapotranspiration. Positive values indicate wetter conditions.",
  aridity_index: "Ratio between precipitation and PET. It summarizes moisture limitation.",
  seasonal_contrast: "Difference or ratio between two selected layers, often two seasons or periods.",
  snow_persistence_ratio: "Snow observation days divided by valid observation days."
};

const maskingHelp: Record<string, string> = {
  binary_threshold_mask: "Creates a 0/1 mask from a numeric threshold.",
  class_mask: "Creates a 0/1 mask for one categorical class value."
};

const terrainHelp: Record<string, string> = {
  slope: "Slope in degrees from a DEM. Requires an elevation/DEM layer.",
  aspect: "Downslope orientation in degrees from north. Requires an elevation/DEM layer.",
  ruggedness: "Local elevation variability around each cell. Requires a DEM.",
  tpi: "Topographic Position Index: cell elevation minus local mean elevation.",
  roughness: "Local max minus local min elevation inside the selected radius."
};

const focalHelp: Record<string, string> = {
  mean: "Local moving-window average around each cell.",
  std: "Local moving-window standard deviation around each cell.",
  min: "Local minimum inside the selected radius.",
  max: "Local maximum inside the selected radius.",
  sum: "Local sum inside the selected radius.",
  majority: "Most frequent class inside the selected radius. Best for categorical rasters.",
  diversity: "Number of unique classes inside the selected radius."
};

function DerivedPanel({
  selectedSources,
  catalog,
  derivedFeatures,
  setDerivedFeatures
}: {
  selectedSources: SourceSelection[];
  catalog: WorkbenchCatalog;
  derivedFeatures: DerivedFeatureConfig[];
  setDerivedFeatures: (rows: DerivedFeatureConfig[]) => void;
}) {
  const plannedLayers = useMemo(
    () => buildPlannedLayers(catalog, selectedSources),
    [catalog, selectedSources]
  );
  const derivedInputLayers = useMemo<PlannedLayer[]>(
    () => derivedFeatures.map((feature) => ({
      id: `derived:${feature.name}`,
      label: `Derived · ${feature.name}`,
      sourceTitle: "Derived features",
      query: {
        source_id: "derived",
        variable: feature.name
      },
      variable: feature.name,
      unit: feature.unit,
      valueSemantics: feature.value_semantics
    })),
    [derivedFeatures]
  );
  const availableLayers = useMemo(
    () => [...plannedLayers, ...derivedInputLayers],
    [plannedLayers, derivedInputLayers]
  );
  const [recipe, setRecipe] = useState("thermal_range");
  const [primaryLayerId, setPrimaryLayerId] = useState("");
  const [secondaryLayerId, setSecondaryLayerId] = useState("");
  const [recipeInputs, setRecipeInputs] = useState<Record<string, string>>({});
  const [recipeOutputName, setRecipeOutputName] = useState("");
  const [maskRecipe, setMaskRecipe] = useState("binary_threshold_mask");
  const [maskLayerId, setMaskLayerId] = useState("");
  const [maskOutputName, setMaskOutputName] = useState("");
  const [maskThreshold, setMaskThreshold] = useState(0);
  const [maskClassValue, setMaskClassValue] = useState(1);
  const [expression, setExpression] = useState("x - y");
  const [expressionLayerIds, setExpressionLayerIds] = useState<Record<string, string>>({});
  const [expressionOutputName, setExpressionOutputName] = useState("");
  const [terrainOutputName, setTerrainOutputName] = useState("");
  const [focalOutputName, setFocalOutputName] = useState("");
  const [distanceOutputName, setDistanceOutputName] = useState("");
  const [distanceLayerId, setDistanceLayerId] = useState("");
  const [unit, setUnit] = useState("");
  const [valueSemantics, setValueSemantics] = useState("intensive");
  const [description, setDescription] = useState("");
  const [terrainMethod, setTerrainMethod] = useState("slope");
  const [focalMethod, setFocalMethod] = useState("mean");
  const [radius, setRadius] = useState(1);
  const [classValue, setClassValue] = useState(1);

  const layerById = useMemo(
    () => new Map(availableLayers.map((layer) => [layer.id, layer])),
    [availableLayers]
  );
  const firstLayer = availableLayers[0];
  const primaryLayer = layerById.get(primaryLayerId) ?? firstLayer;
  const secondaryLayer = layerById.get(secondaryLayerId) ?? availableLayers[1] ?? firstLayer;
  const demLayers = availableLayers.filter(isDemLayer);
  const numericLayers = availableLayers.filter(isNumericLayer);
  const categoricalLayers = availableLayers.filter(isCategoricalLayer);
  const distanceLayers = availableLayers.filter((layer) => isBinaryLayer(layer) || isCategoricalLayer(layer));
  const terrainLayer = demLayers.find((layer) => layer.id === primaryLayerId) ?? demLayers[0];
  const focalCandidateLayers = ["majority", "diversity"].includes(focalMethod) && categoricalLayers.length > 0
    ? categoricalLayers
    : availableLayers;
  const focalLayer = focalCandidateLayers.find((layer) => layer.id === primaryLayerId) ?? focalCandidateLayers[0];
  const distanceLayer = distanceLayers.find((layer) => layer.id === distanceLayerId) ?? distanceLayers[0];
  const maskCandidateLayers = maskRecipe === "binary_threshold_mask"
    ? numericLayers
    : categoricalLayers;
  const maskLayer = maskCandidateLayers.find((layer) => layer.id === maskLayerId) ?? maskCandidateLayers[0];

  function findLayer(variable: string, sameAs?: PlannedLayer) {
    return availableLayers.find((layer) =>
      layer.variable === variable &&
      sameLayerContext(layer, sameAs)
    );
  }

  function findMatchingLayer(names: string[], sameAs?: PlannedLayer) {
    return availableLayers.find((layer) => layerVariableMatches(layer, names) && sameLayerContext(layer, sameAs));
  }

  function addFeature(feature: DerivedFeatureConfig) {
    setDerivedFeatures([...derivedFeatures, feature]);
    setDescription("");
  }

  function layerFromRecipeInput(alias: string, candidates: PlannedLayer[], fallback?: PlannedLayer) {
    const selected = layerById.get(recipeInputs[alias] ?? "");
    if (selected && candidates.some((layer) => layer.id === selected.id)) return selected;
    if (fallback && candidates.some((layer) => layer.id === fallback.id)) return fallback;
    return candidates[0];
  }

  function recipeInputSelect(alias: string, label: string, candidates: PlannedLayer[], fallback?: PlannedLayer, emptyText?: string) {
    const fallbackAllowed = fallback && candidates.some((layer) => layer.id === fallback.id);
    const currentValue = candidates.some((layer) => layer.id === recipeInputs[alias])
      ? recipeInputs[alias]
      : "";
    return (
      <label>
        {label}
        <select
          value={currentValue}
          disabled={candidates.length === 0}
          onChange={(event) => setRecipeInputs({ ...recipeInputs, [alias]: event.target.value })}
        >
          {fallbackAllowed && <option value="">Auto: {fallback.label}</option>}
          {!fallbackAllowed && <option value="">{candidates.length === 0 ? "No valid layers" : "Select layer"}</option>}
          {candidates.map((layer) => (
            <option key={layer.id} value={layer.id}>{layer.label}</option>
          ))}
        </select>
        {candidates.length === 0 && (
          <small className="field-hint">{emptyText ?? "No selected layer matches this input requirement."}</small>
        )}
      </label>
    );
  }

  function recipeCandidates(alias: string) {
    if (recipe === "thermal_range") {
      return alias === "tmax"
        ? availableLayers.filter((layer) => layerVariableMatches(layer, ["tmax", "tasmax", "maximum_temperature"]))
        : availableLayers.filter((layer) => layerVariableMatches(layer, ["tmin", "tasmin", "minimum_temperature"]));
    }
    if (recipe === "water_balance" || recipe === "aridity_index") {
      return alias === "prec"
        ? availableLayers.filter((layer) => layerVariableMatches(layer, ["prec", "precipitation", "ppt"]))
        : availableLayers.filter((layer) => layerVariableMatches(layer, ["pet", "eto", "evapotranspiration"]));
    }
    if (recipe === "snow_persistence_ratio") {
      return alias === "snow_days"
        ? availableLayers.filter((layer) => layerVariableMatches(layer, ["snow_days", "snow_days_count", "snow"]))
        : availableLayers.filter((layer) => layerVariableMatches(layer, ["valid_days", "valid_observations", "valid"]));
    }
    if (recipe === "seasonal_contrast") {
      return numericLayers;
    }
    return availableLayers;
  }

  function addGuidedRecipe() {
    if (!primaryLayer) return;

    let inputs: Record<string, DerivedInputQuery> = {};
    let parameters: Record<string, unknown> = {};
    let defaultUnit = unit || primaryLayer.unit || "unitless";
    let defaultExpressionName = recipe;

    if (recipe === "thermal_range") {
      const tmaxCandidates = recipeCandidates("tmax");
      const tminCandidates = recipeCandidates("tmin");
      const tmax = findLayer("tmax", primaryLayer) ?? findMatchingLayer(["tmax", "tasmax", "maximum_temperature"], primaryLayer) ?? tmaxCandidates[0];
      const tmin = findLayer("tmin", tmax ?? primaryLayer) ?? findMatchingLayer(["tmin", "tasmin", "minimum_temperature"], tmax ?? primaryLayer) ?? tminCandidates[0];
      const selectedTmax = layerFromRecipeInput("tmax", tmaxCandidates, tmax);
      const selectedTmin = layerFromRecipeInput("tmin", tminCandidates, tmin);
      if (!selectedTmax || !selectedTmin) return;
      inputs = { tmax: selectedTmax.query, tmin: selectedTmin.query };
      defaultUnit = unit || "degC";
    } else if (recipe === "water_balance" || recipe === "aridity_index") {
      const precCandidates = recipeCandidates("prec");
      const petCandidates = recipeCandidates("pet");
      const prec = findLayer("prec", primaryLayer) ?? findMatchingLayer(["prec", "precipitation", "ppt"], primaryLayer) ?? precCandidates[0];
      const pet = findLayer("pet", prec ?? primaryLayer) ?? findMatchingLayer(["pet", "eto", "evapotranspiration"], prec ?? primaryLayer) ?? petCandidates[0];
      const selectedPrec = layerFromRecipeInput("prec", precCandidates, prec);
      const selectedPet = layerFromRecipeInput("pet", petCandidates, pet);
      if (!selectedPrec || !selectedPet) return;
      inputs = { prec: selectedPrec.query, pet: selectedPet.query };
      defaultUnit = recipe === "aridity_index" ? "ratio" : (unit || "mm");
      parameters = recipe === "aridity_index" ? { convention: "prec_over_pet" } : {};
    } else if (recipe === "snow_persistence_ratio") {
      const snowCandidates = recipeCandidates("snow_days");
      const validCandidates = recipeCandidates("valid_days");
      const snow = findLayer("snow_days", primaryLayer) ?? findMatchingLayer(["snow_days", "snow_days_count", "snow"], primaryLayer) ?? snowCandidates[0];
      const valid = findLayer("valid_days", snow) ?? findMatchingLayer(["valid_days", "valid_observations", "valid"], snow) ?? validCandidates[0];
      const selectedSnow = layerFromRecipeInput("snow_days", snowCandidates, snow);
      const selectedValid = layerFromRecipeInput("valid_days", validCandidates, valid);
      if (!selectedSnow || !selectedValid) return;
      inputs = { snow_days: selectedSnow.query, valid_days: selectedValid.query };
      defaultUnit = "ratio";
    } else if (recipe === "seasonal_contrast") {
      const candidates = recipeCandidates("a");
      const a = layerFromRecipeInput("a", candidates, candidates.find((layer) => layer.id === primaryLayer?.id) ?? candidates[0]);
      const b = layerFromRecipeInput("b", candidates, candidates.find((layer) => layer.id === secondaryLayer?.id) ?? candidates[1] ?? candidates[0]);
      if (!a || !b) return;
      inputs = { a: a.query, b: b.query };
      parameters = { metric: "difference" };
      defaultUnit = unit || a.unit || "source_units";
    }

    addFeature({
      name: sanitizeDerivedName(recipeOutputName || defaultExpressionName),
      operation: "recipe",
      recipe,
      description: description || humanizeId(recipeOutputName || defaultExpressionName),
      unit: defaultUnit,
      value_semantics: recipe.includes("mask") ? "categorical" : valueSemantics,
      output_dtype: recipe.includes("mask") ? "uint8" : "float32",
      parameters,
      inputs
    });
    setRecipeOutputName("");
  }

  function addMaskingFeature() {
    if (!maskLayer) return;
    const isThreshold = maskRecipe === "binary_threshold_mask";
    addFeature({
      name: sanitizeDerivedName(
        maskOutputName ||
          (isThreshold
            ? `${maskLayer.variable}_threshold_mask`
            : `${maskLayer.variable}_class_${maskClassValue}_mask`)
      ),
      operation: "recipe",
      recipe: maskRecipe,
      description: description || (isThreshold
        ? `Binary threshold mask derived from ${maskLayer.label}.`
        : `Class mask derived from ${maskLayer.label}.`),
      unit: "binary",
      value_semantics: "categorical",
      output_dtype: "uint8",
      parameters: isThreshold
        ? { operator: ">=", threshold: maskThreshold }
        : { class_value: maskClassValue },
      inputs: {
        x: maskLayer.query
      }
    });
    setMaskOutputName("");
  }

  function addExpression() {
    const xLayer = layerById.get(expressionLayerIds.x ?? "") ?? primaryLayer;
    const yLayer = layerById.get(expressionLayerIds.y ?? "") ?? secondaryLayer;
    const zLayer = layerById.get(expressionLayerIds.z ?? "");
    if (!xLayer) return;
    const inputs: Record<string, DerivedInputQuery> = { x: xLayer.query };
    if (yLayer) inputs.y = yLayer.query;
    if (zLayer) inputs.z = zLayer.query;
    addFeature({
      name: sanitizeDerivedName(expressionOutputName || "custom_expression"),
      operation: "expression",
      expression,
      description: description || "Custom derived raster expression.",
      unit: unit || xLayer.unit || "unitless",
      value_semantics: valueSemantics,
      output_dtype: "float32",
      inputs
    });
    setExpressionOutputName("");
  }

  function addSpatialOperation(operation: "terrain" | "focal" | "distance") {
    const inputLayer = operation === "terrain"
      ? terrainLayer
      : operation === "distance"
        ? distanceLayer
        : focalLayer;
    if (!inputLayer) return;
    if (operation === "terrain" && !isDemLayer(inputLayer)) return;
    const spatialMethod = operation === "terrain" ? terrainMethod : operation === "focal" ? focalMethod : "distance_to_mask";
    const nameFromState = operation === "terrain"
      ? terrainOutputName
      : operation === "focal"
        ? focalOutputName
        : distanceOutputName;
    addFeature({
      name: sanitizeDerivedName(nameFromState || `${inputLayer.variable}_${spatialMethod}`),
      operation,
      method: spatialMethod,
      description: description || `${humanizeId(spatialMethod)} derived from ${inputLayer.label}.`,
      unit: operation === "distance" ? "m" : unit || inputLayer.unit || "unitless",
      value_semantics: operation === "distance" ? "intensive" : valueSemantics,
      output_dtype: "float32",
      parameters: {
        radius: operation === "focal" || ["ruggedness", "tpi", "roughness"].includes(spatialMethod) ? radius : undefined,
        class_value: operation === "distance" ? classValue : undefined
      },
      inputs: {
        [operation === "terrain" ? "dem" : operation === "distance" ? "mask" : "x"]: inputLayer.query
      }
    });
    if (operation === "terrain") setTerrainOutputName("");
    if (operation === "focal") setFocalOutputName("");
    if (operation === "distance") setDistanceOutputName("");
  }

  return (
    <main className="workspace derived-workspace">
      <section className="panel derived-header-panel">
        <div className="panel-head">
          <h2>Derived Features</h2>
          <span className="field-hint">{availableLayers.length} available input layers</span>
        </div>
        {selectedSources.length === 0 && (
          <div className="notice info">Select at least one source before adding derived features.</div>
        )}
        {selectedSources.length > 0 && availableLayers.length === 0 && (
          <div className="notice info">Select variables before adding derived features.</div>
        )}
      </section>

      <section className="panel derived-builder">
        <div className="builder-head">
          <span className="builder-step">01</span>
          <div>
            <h3>Guided recipes</h3>
            <p className="builder-copy">
              Predefined cell-by-cell formulas for common raster derivatives. Choose the main layer,
              and add a secondary layer when the recipe combines two variables.
            </p>
          </div>
          <InfoTip text="Predefined pixel-wise formulas with strict input filters. Thermal range only accepts max/min temperature, water balance needs precipitation/PET, and snow persistence needs snow/valid-day layers." />
        </div>
        <div className="form-grid">
          <label className="span-2">
            <span className="label-line">Recipe <InfoTip text={recipeHelp[recipe] ?? "Derived recipe."} /></span>
            <select value={recipe} onChange={(event) => setRecipe(event.target.value)}>
              <option value="thermal_range">Thermal range</option>
              <option value="water_balance">Water balance</option>
              <option value="aridity_index">Aridity index</option>
              <option value="seasonal_contrast">Seasonal contrast</option>
              <option value="snow_persistence_ratio">Snow persistence ratio</option>
            </select>
            <small className="field-hint">{recipeHelp[recipe]}</small>
          </label>
          {recipe === "thermal_range" && (
            <>
              {recipeInputSelect("tmax", "Maximum temperature input", recipeCandidates("tmax"), findLayer("tmax", primaryLayer) ?? findMatchingLayer(["tmax", "tasmax", "maximum_temperature"], primaryLayer), "No tmax/tasmax layer is selected.")}
              {recipeInputSelect("tmin", "Minimum temperature input", recipeCandidates("tmin"), findLayer("tmin", primaryLayer) ?? findMatchingLayer(["tmin", "tasmin", "minimum_temperature"], primaryLayer), "No tmin/tasmin layer is selected.")}
            </>
          )}
          {(recipe === "water_balance" || recipe === "aridity_index") && (
            <>
              {recipeInputSelect("prec", "Precipitation input", recipeCandidates("prec"), findLayer("prec", primaryLayer) ?? findMatchingLayer(["prec", "precipitation", "ppt"], primaryLayer), "No precipitation layer is selected.")}
              {recipeInputSelect("pet", "PET input", recipeCandidates("pet"), findLayer("pet", primaryLayer) ?? findMatchingLayer(["pet", "eto", "evapotranspiration"], primaryLayer), "No PET/evapotranspiration layer is selected.")}
            </>
          )}
          {recipe === "seasonal_contrast" && (
            <>
              {recipeInputSelect("a", "First input", recipeCandidates("a"), primaryLayer && isNumericLayer(primaryLayer) ? primaryLayer : numericLayers[0], "No numeric layer is selected.")}
              {recipeInputSelect("b", "Second input", recipeCandidates("b"), secondaryLayer && isNumericLayer(secondaryLayer) ? secondaryLayer : numericLayers[1] ?? numericLayers[0], "No second numeric layer is selected.")}
            </>
          )}
          {recipe === "snow_persistence_ratio" && (
            <>
              {recipeInputSelect("snow_days", "Snow-days input", recipeCandidates("snow_days"), findLayer("snow_days", primaryLayer) ?? findMatchingLayer(["snow_days", "snow_days_count", "snow"], primaryLayer), "No snow-days layer is selected.")}
              {recipeInputSelect("valid_days", "Valid-days input", recipeCandidates("valid_days"), findLayer("valid_days", primaryLayer) ?? findMatchingLayer(["valid_days", "valid_observations", "valid"], primaryLayer), "No valid-days layer is selected.")}
            </>
          )}
          <label>
            Output name
            <input value={recipeOutputName} onChange={(event) => setRecipeOutputName(event.target.value)} placeholder="auto if blank" />
          </label>
        </div>
        <button className="primary" onClick={addGuidedRecipe} disabled={!primaryLayer}>Add guided feature</button>
      </section>

      <section className="panel derived-builder masking-builder">
        <div className="builder-head">
          <span className="builder-step">02</span>
          <div>
            <h3>Maskings</h3>
            <p className="builder-copy">
              Build binary masks from numeric thresholds or categorical classes before using them
              in distance and focal operations.
            </p>
          </div>
          <InfoTip text="Threshold masks need numeric layers. Class masks need categorical or binary layers. Outputs are 0/1 rasters." />
        </div>
        <div className="form-grid">
          <label>
            Mask type
            <select value={maskRecipe} onChange={(event) => setMaskRecipe(event.target.value)}>
              <option value="binary_threshold_mask">Binary threshold mask</option>
              <option value="class_mask">Class mask</option>
            </select>
            <small className="field-hint">{maskingHelp[maskRecipe]}</small>
          </label>
          <label>
            Input layer
            <select
              value={maskLayer?.id ?? ""}
              disabled={maskCandidateLayers.length === 0}
              onChange={(event) => setMaskLayerId(event.target.value)}
            >
              {maskCandidateLayers.length === 0 && <option value="">No valid layers</option>}
              {maskCandidateLayers.map((layer) => (
                <option key={layer.id} value={layer.id}>{layer.label}</option>
              ))}
            </select>
            {maskCandidateLayers.length === 0 && (
              <small className="field-hint">
                {maskRecipe === "binary_threshold_mask"
                  ? "Select a numeric layer before creating a threshold mask."
                  : "Select a categorical or binary layer before creating a class mask."}
              </small>
            )}
          </label>
          {maskRecipe === "binary_threshold_mask" && (
            <label>
              Threshold
              <input type="number" value={maskThreshold} onChange={(event) => setMaskThreshold(Number(event.target.value))} />
            </label>
          )}
          {maskRecipe === "class_mask" && (
            <label>
              Class value
              <input type="number" value={maskClassValue} onChange={(event) => setMaskClassValue(Number(event.target.value))} />
            </label>
          )}
          <label>
            Output name
            <input value={maskOutputName} onChange={(event) => setMaskOutputName(event.target.value)} placeholder="auto if blank" />
          </label>
        </div>
        <button className="primary" onClick={addMaskingFeature} disabled={!maskLayer}>Add masking feature</button>
      </section>

      <section className="panel derived-builder spatial-builder">
        <div className="builder-head">
          <span className="builder-step">03</span>
          <div>
            <h3>Spatial operations</h3>
            <p className="builder-copy">
              Spatial derivatives use terrain shape, moving windows or distance context. Terrain
              methods need a DEM/elevation layer; focal and distance methods can use other selected rasters.
            </p>
          </div>
          <InfoTip text="Terrain methods require a DEM/elevation layer. Focal methods summarize nearby cells. Distance methods need a binary mask or a categorical class value." />
        </div>
        <div className="spatial-method-grid">
          <div className="spatial-method-card">
            <div className="method-card-head">
              <strong>DEM terrain</strong>
              <InfoTip text="Slope, aspect and terrain indices require a selected elevation/DEM layer." />
            </div>
            <p className="method-copy">
              Use this only when the main input is an elevation raster. Slope and aspect are direct
              terrain derivatives; ruggedness, TPI and roughness use the radius window below.
            </p>
            <label>
              DEM input
              <select
                value={terrainLayer?.id ?? ""}
                disabled={demLayers.length === 0}
                onChange={(event) => setPrimaryLayerId(event.target.value)}
              >
                {demLayers.length === 0 && <option value="">No DEM layer selected</option>}
                {demLayers.map((layer) => (
                  <option key={layer.id} value={layer.id}>{layer.label}</option>
                ))}
              </select>
            </label>
            <label>
              Terrain method
              <select value={terrainMethod} onChange={(event) => setTerrainMethod(event.target.value)}>
                <option value="slope">Slope</option>
                <option value="aspect">Aspect</option>
                <option value="ruggedness">Ruggedness</option>
                <option value="tpi">TPI</option>
                <option value="roughness">Roughness</option>
              </select>
            </label>
            <label>
              Output name
              <input value={terrainOutputName} onChange={(event) => setTerrainOutputName(event.target.value)} placeholder="auto if blank" />
            </label>
            {["ruggedness", "tpi", "roughness"].includes(terrainMethod) && (
              <label>
                <span className="label-line">Radius in cells <InfoTip text="Radius 1 means a 3 by 3 cell window." /></span>
                <input type="number" min={1} value={radius} onChange={(event) => setRadius(Number(event.target.value))} />
              </label>
            )}
            <small className="field-hint">{terrainHelp[terrainMethod]}</small>
            {demLayers.length === 0 && (
              <div className="notice info compact-notice">
                Select a DEM/elevation variable before adding terrain-derived features.
              </div>
            )}
            <button onClick={() => addSpatialOperation("terrain")} disabled={!terrainLayer}>
              Add terrain
            </button>
          </div>

          <div className="spatial-method-card">
            <div className="method-card-head">
              <strong>Focal window</strong>
              <InfoTip text="Focal operations summarize values in a moving window around each cell." />
            </div>
            <p className="method-copy">
              Use this for local context around each pixel, such as neighborhood mean, variation,
              dominant class or class diversity.
            </p>
            <label>
              Focal method
              <select value={focalMethod} onChange={(event) => setFocalMethod(event.target.value)}>
                <option value="mean">Focal mean</option>
                <option value="std">Focal std</option>
                <option value="min">Focal min</option>
                <option value="max">Focal max</option>
                <option value="sum">Focal sum</option>
                <option value="majority">Focal majority</option>
                <option value="diversity">Focal diversity</option>
              </select>
            </label>
            <label>
              Input layer
              <select
                value={focalLayer?.id ?? ""}
                disabled={focalCandidateLayers.length === 0}
                onChange={(event) => setPrimaryLayerId(event.target.value)}
              >
                {focalCandidateLayers.length === 0 && <option value="">No valid layer selected</option>}
                {focalCandidateLayers.map((layer) => (
                  <option key={layer.id} value={layer.id}>{layer.label}</option>
                ))}
              </select>
            </label>
            <label>
              <span className="label-line">Radius in cells <InfoTip text="Radius 1 means a 3 by 3 cell window." /></span>
              <input type="number" min={1} value={radius} onChange={(event) => setRadius(Number(event.target.value))} />
            </label>
            <label>
              Output name
              <input value={focalOutputName} onChange={(event) => setFocalOutputName(event.target.value)} placeholder="auto if blank" />
            </label>
            <small className="field-hint">{focalHelp[focalMethod]}</small>
            <button onClick={() => addSpatialOperation("focal")} disabled={!focalLayer}>Add focal</button>
          </div>

          <div className="spatial-method-card">
            <div className="method-card-head">
              <strong>Distance</strong>
              <InfoTip text="Distance-to computes metres from each cell to the nearest positive mask/class cell." />
            </div>
            <p className="method-copy">
              Use this after selecting a binary mask, or choose a class value from a categorical raster.
              The output is distance in metres to the nearest matching cell.
            </p>
            <label>
              Distance-to input
              <select
                value={distanceLayer?.id ?? ""}
                disabled={distanceLayers.length === 0}
                onChange={(event) => setDistanceLayerId(event.target.value)}
              >
                {distanceLayers.length === 0 && <option value="">No binary/categorical layer selected</option>}
                {distanceLayers.map((layer) => (
                  <option key={layer.id} value={layer.id}>{layer.label}</option>
                ))}
              </select>
            </label>
            <label>
              Class value
              <input type="number" value={classValue} onChange={(event) => setClassValue(Number(event.target.value))} />
            </label>
            <label>
              Output name
              <input value={distanceOutputName} onChange={(event) => setDistanceOutputName(event.target.value)} placeholder="auto if blank" />
            </label>
            <small className="field-hint">Use a binary mask or select a class value from a categorical raster.</small>
            {distanceLayers.length === 0 && (
              <div className="notice info compact-notice">
                Distance inputs must be binary masks or categorical rasters.
              </div>
            )}
            <button onClick={() => addSpatialOperation("distance")} disabled={!distanceLayer}>Add distance-to</button>
          </div>
        </div>
      </section>

      <section className="panel derived-builder advanced-expression-builder">
        <div className="builder-head">
          <span className="builder-step">04</span>
          <div>
            <h3>Advanced expression</h3>
            <p className="builder-copy">
              Write a custom raster expression using selected inputs as aliases. Use this for
              formulas that are not covered by the guided recipes.
            </p>
          </div>
          <InfoTip text="Custom map algebra. Use x, y and z as selected inputs. Safe functions include where, sqrt, log, minimum and maximum." />
        </div>
        <div className="form-grid">
          <label className="span-2">
            Expression
            <input value={expression} onChange={(event) => setExpression(event.target.value)} />
            <small className="field-hint">Example: where(x &gt; 0, x / maximum(y, 1), nan). Use x, y and z as the selected aliases below.</small>
          </label>
          {(["x", "y", "z"] as const).map((alias) => (
            <label key={alias}>
              {alias} input
              <select
                value={expressionLayerIds[alias] ?? ""}
                disabled={availableLayers.length === 0}
                onChange={(event) =>
                  setExpressionLayerIds({ ...expressionLayerIds, [alias]: event.target.value })
                }
              >
                <option value="">{alias === "x" ? `Auto: ${primaryLayer?.label ?? "select layer"}` : "Unused"}</option>
                {availableLayers.map((layer) => (
                  <option key={layer.id} value={layer.id}>{layer.label}</option>
                ))}
              </select>
            </label>
          ))}
          <label>
            Output name
            <input value={expressionOutputName} onChange={(event) => setExpressionOutputName(event.target.value)} placeholder="custom_expression" />
          </label>
          <label>
            Unit
            <input value={unit} onChange={(event) => setUnit(event.target.value)} placeholder="degC, mm, ratio..." />
          </label>
          <label>
            Value semantics
            <select value={valueSemantics} onChange={(event) => setValueSemantics(event.target.value)}>
              {(catalog.value_semantics ?? ["intensive", "percentage", "categorical"]).map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
          </label>
          <label className="span-2">
            Description
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} rows={2} />
          </label>
        </div>
        <button className="primary" onClick={addExpression} disabled={!primaryLayer}>Add expression feature</button>
      </section>

      <section className="panel derived-summary-panel">
        <div className="panel-head">
          <h2>Selected derived layers</h2>
          <span className="field-hint">{derivedFeatures.length} configured</span>
        </div>
        <div className="aggregation-list">
          {derivedFeatures.map((feature, index) => (
            <div className="aggregation-chip" key={`${feature.name}-${index}`}>
              <span>
                <strong>{feature.name}</strong>
                <small>{feature.operation}{feature.recipe ? ` · ${feature.recipe}` : ""}{feature.method ? ` · ${feature.method}` : ""}</small>
              </span>
              <button className="ghost danger" onClick={() => setDerivedFeatures(derivedFeatures.filter((_, itemIndex) => itemIndex !== index))}>Remove</button>
            </div>
          ))}
          {derivedFeatures.length === 0 && (
            <div className="empty-state">No derived layers selected yet.</div>
          )}
        </div>
      </section>
    </main>
  );
}

function ReviewPanel({
  yamlText,
  validation,
  apiError,
  saveStatus,
  validate,
  renderFromServer,
  copyYaml,
  saveYamlToRuns,
  downloadYaml,
  plannedLayers,
  derivedFeatures,
  removePlannedLayer,
  removeDerivedFeature
}: {
  yamlText: string;
  validation: ValidationReport | null;
  apiError: string | null;
  saveStatus: string | null;
  validate: () => void;
  renderFromServer: () => void;
  copyYaml: () => void;
  saveYamlToRuns: () => void;
  downloadYaml: () => void;
  plannedLayers: PlannedLayer[];
  derivedFeatures: DerivedFeatureConfig[];
  removePlannedLayer: (layer: PlannedLayer) => void;
  removeDerivedFeature: (index: number) => void;
}) {
  return (
    <main className="workspace review-grid">
      <section className="panel">
        <div className="panel-head">
          <h2>Review</h2>
          <div className="button-row">
            <button className="primary" onClick={validate}>Validate</button>
            <button onClick={renderFromServer}>Render</button>
            <button onClick={copyYaml}>Copy YAML</button>
            <button onClick={saveYamlToRuns}>Save YAML</button>
            <button onClick={downloadYaml}>Download YAML</button>
          </div>
        </div>
        {apiError && <div className="notice error">{apiError}</div>}
        {saveStatus && <div className="notice success">{saveStatus}</div>}
        {validation && (
          <div className={`notice ${validation.ok ? "success" : "error"}`}>
            {validation.ok ? "Valid config" : "Invalid config"} · {validation.estimated_layers} estimated layers
          </div>
        )}
        {validation?.errors.map((error) => (
          <div className="notice error" key={error}>{error}</div>
        ))}
        {validation?.warnings.map((warning) => (
          <div className="notice info" key={warning}>{warning}</div>
        ))}
        {validation?.sources.map((source) => (
          <div className="summary-row" key={source.id}>
            <strong>{source.id}</strong>
            <span>{source.estimated_layers} layers</span>
          </div>
        ))}
      </section>

      <section className="panel review-layers-panel">
        <div className="panel-head">
          <h2>Planned Layers</h2>
          <span className="field-hint">{plannedLayers.length + derivedFeatures.length} selected outputs</span>
        </div>
        <div className="review-layer-list">
          {plannedLayers.map((layer) => (
            <div className="review-layer-row" key={layer.id}>
              <span>
                <strong>{layer.label}</strong>
                <small>{layer.sourceTitle}{layer.valueSemantics ? ` · ${layer.valueSemantics}` : ""}</small>
              </span>
              <button className="ghost danger" onClick={() => removePlannedLayer(layer)}>Remove</button>
            </div>
          ))}
          {derivedFeatures.map((feature, index) => (
            <div className="review-layer-row derived-row" key={`${feature.name}-${index}`}>
              <span>
                <strong>{feature.name}</strong>
                <small>{feature.operation}{feature.recipe ? ` · ${feature.recipe}` : ""}{feature.method ? ` · ${feature.method}` : ""}</small>
              </span>
              <button className="ghost danger" onClick={() => removeDerivedFeature(index)}>Remove</button>
            </div>
          ))}
          {plannedLayers.length === 0 && derivedFeatures.length === 0 && (
            <div className="empty-state">No layers selected yet.</div>
          )}
        </div>
      </section>

      <section className="panel yaml-panel">
        <h2>YAML</h2>
        <pre>{yamlText}</pre>
      </section>
    </main>
  );
}

export default App;
