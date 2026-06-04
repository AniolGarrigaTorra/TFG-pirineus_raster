import { useEffect, useMemo, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { createAoiConfig, createProjectGrid, fetchCatalog, renderRunConfig, saveRunConfig, validateRunConfig } from "./api";
import type {
  AoiCatalog,
  AoiBounds,
  CustomAggregation,
  DerivedFeatureConfig,
  DerivedInputQuery,
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
type StartMode = "menu" | "project" | "aoi" | "sources";
const defaultBackgroundUrl = "/backgrounds/pirineus-background.png";
const backgroundManifestUrl = "/backgrounds/manifest.json";
const backgroundRotationMs = 5 * 60 * 1000;
const temporalDimensionKeys = new Set(["year", "years", "month", "months", "season", "seasons"]);
const pyreneesWgs84Envelope = { xmin: -2.8, xmax: 3.9, ymin: 41.0, ymax: 43.9 };
const aoiInitialMapZoom = 8;
const aoiMapViewport = { width: 1180, height: 690 };
type LonLatPoint = { lon: number; lat: number };
type ProjectedPoint = { x: number; y: number };

function normalizeBackgroundUrls(value: unknown) {
  if (!Array.isArray(value)) return [defaultBackgroundUrl];

  const urls = value
    .filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    .map((item) => item.trim());

  return urls.length > 0 ? [...new Set(urls)] : [defaultBackgroundUrl];
}

function pickRandomBackground(urls: string[], currentUrl?: string) {
  if (urls.length === 0) return defaultBackgroundUrl;
  if (urls.length === 1) return urls[0];

  let nextUrl = currentUrl;
  for (let attempt = 0; attempt < 8 && nextUrl === currentUrl; attempt += 1) {
    nextUrl = urls[Math.floor(Math.random() * urls.length)];
  }
  return nextUrl ?? urls[0];
}

function cssBackgroundImage(url: string) {
  return `url("${url.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}")`;
}

function BackgroundCredit() {
  return <div className="photo-credit">Photo: Felipe Valladares</div>;
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

function toggleValue(values: string[], value: string) {
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
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [serverYaml, setServerYaml] = useState("");
  const [apiError, setApiError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [hasStarted, setHasStarted] = useState(false);
  const [startMode, setStartMode] = useState<StartMode>("menu");
  const [backgroundUrls, setBackgroundUrls] = useState<string[]>([defaultBackgroundUrl]);
  const [backgroundUrl, setBackgroundUrl] = useState(defaultBackgroundUrl);

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
        setBackgroundUrl(pickRandomBackground(images));
      })
      .catch(() => {
        if (cancelled) return;
        setBackgroundUrls([defaultBackgroundUrl]);
        setBackgroundUrl(defaultBackgroundUrl);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    document.documentElement.style.setProperty(
      "--pirineus-background-image",
      cssBackgroundImage(backgroundUrl)
    );

    return () => {
      document.documentElement.style.removeProperty("--pirineus-background-image");
    };
  }, [backgroundUrl]);

  useEffect(() => {
    if (backgroundUrls.length <= 1) return undefined;

    const intervalId = window.setInterval(() => {
      setBackgroundUrl((current) => pickRandomBackground(backgroundUrls, current));
    }, backgroundRotationMs);

    return () => window.clearInterval(intervalId);
  }, [backgroundUrls]);

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
    const sourceEntries = selectedSources.map((selection) => {
      const source = catalog?.sources.find((item) => item.id === selection.id);
      const variableNames = selection.variables.filter((name) =>
        (source?.variables ?? []).some((variable) => variable.name === name)
      );
      const layerNames = selection.variables.filter((name) =>
        (source?.layers ?? []).some((layer) => layer.name === name)
      );
      const dimensions = Object.fromEntries(
        Object.entries(selection.dimensions).filter(
          ([key]) => !temporalDimensionKeys.has(key.toLowerCase())
        )
      );

      const temporal = buildTemporalConfig(source, selection);
      const resamplingOverrides = Object.keys(selection.resamplingByVariable).length > 0
        ? {
            by_variable: selection.resamplingByVariable
          }
        : undefined;
      const processingOverrides = selection.sourceResolution && selection.sourceResolution !== source?.source_resolution
        ? {
            source_resolution: selection.sourceResolution
          }
        : undefined;
      const downloadOverrides = selection.keepRawAfterClip !== (source?.keep_raw_after_clip_default ?? true)
        ? {
            keep_raw_after_clip: selection.keepRawAfterClip
          }
        : undefined;
      const overrides = {
        resampling: resamplingOverrides,
        processing: processingOverrides,
        download: downloadOverrides
      };
      const hasOverrides = Object.values(overrides).some((value) => value !== undefined);
      const select = {
        variables: (source?.variables ?? []).length > 0 || selection.categoryFractions.length > 0
          ? variableNames
          : undefined,
        layers: layerNames.length > 0 ? layerNames : undefined,
        category_fractions: selection.categoryFractions.length > 0
          ? selection.categoryFractions.map((item) => ({
              variable: item.variable,
              name: item.name,
              class_values: item.class_values,
              label: item.label
            }))
          : undefined,
        dimensions: Object.keys(dimensions).length > 0 ? dimensions : undefined,
        temporal
      };
      const hasSelect = Object.values(select).some((value) => value !== undefined);

      return {
        id: selection.id,
        config: selection.config,
        stages: selection.stages.length > 0 ? selection.stages : undefined,
        select: hasSelect ? select : undefined,
        overrides: hasOverrides ? overrides : undefined
      };
    });

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
      sources: sourceEntries,
      derived_features: derivedFeatures.length > 0 ? derivedFeatures : undefined,
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
    derivedFeatures
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

  if (!hasStarted) {
    return (
      <main className="home-screen">
        <section className="home-hero">
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
      <div className="app-shell workbench-shell">
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
      </div>
    );
  }

  if (startMode === "aoi") {
    return (
      <div className="app-shell workbench-shell">
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
      </div>
    );
  }

  if (startMode === "sources") {
    return (
      <div className="app-shell workbench-shell">
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
      </div>
    );
  }

  return (
    <div className="app-shell workbench-shell">
      <header className="topbar">
        <div>
          <h1>Pirineus Raster Workbench</h1>
          <p>{targetCrs} · {selectedSources.length} sources · {validation?.estimated_layers ?? 0} estimated layers</p>
        </div>
        <div className="topbar-actions">
          <button className="ghost" onClick={() => setStartMode("menu")}>Home</button>
          <div className={`api-pill ${catalogError ? "bad" : catalog ? "good" : "loading"}`}>
            {apiStatus}
          </div>
        </div>
      </header>

      <nav className="tabs" aria-label="Workbench sections">
        {tabs.map((tab) => (
          <button
            key={tab}
            className={activeTab === tab ? "active" : ""}
            onClick={() => setActiveTab(tab)}
          >
            <span aria-hidden="true">{tabs.indexOf(tab) + 1}</span>
            {tab}
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

      {activeTab === "Project" && catalog && (
        <ProjectPanel
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
        />
      )}

      {activeTab === "Sources" && catalog && (
        <SourcePanel
          catalog={catalog}
          selections={selections}
          patchSelection={patchSelection}
          setActiveSourceId={setActiveSourceId}
          setActiveTab={setActiveTab}
        />
      )}

      {activeTab === "Variables" && catalog && activeSource && (
        <VariablesPanel
          catalog={catalog}
          sources={selectedCatalogSources}
          source={activeSource}
          selection={selections[activeSource.id]}
          activeSourceId={activeSourceId}
          setActiveSourceId={setActiveSourceId}
          patchSelectionMut={patchSelectionMut}
        />
      )}

      {activeTab === "Variables" && catalog && !activeSource && (
        <NoSelectedSourcesPanel title="Variables" />
      )}

      {activeTab === "Temporal" && catalog && activeSource && (
        <TemporalPanel
          catalog={catalog}
          sources={selectedCatalogSources}
          source={activeSource}
          selection={selections[activeSource.id]}
          activeSourceId={activeSourceId}
          setActiveSourceId={setActiveSourceId}
          patchSelectionMut={patchSelectionMut}
        />
      )}

      {activeTab === "Temporal" && catalog && !activeSource && (
        <NoSelectedSourcesPanel title="Temporal" />
      )}

      {activeTab === "Derived" && catalog && (
        <DerivedPanel
          selectedSources={selectedSources}
          catalog={catalog}
          derivedFeatures={derivedFeatures}
          setDerivedFeatures={setDerivedFeatures}
        />
      )}

      {activeTab === "Review" && (
        <ReviewPanel
          yamlText={yamlText}
          validation={validation}
          apiError={apiError}
          saveStatus={saveStatus}
          validate={validate}
          renderFromServer={renderFromServer}
          copyYaml={copyYaml}
          saveYamlToRuns={saveYamlToRuns}
          downloadYaml={downloadYaml}
          plannedLayers={plannedReviewLayers}
          derivedFeatures={derivedFeatures}
          removePlannedLayer={removePlannedLayer}
          removeDerivedFeature={(index) =>
            setDerivedFeatures(derivedFeatures.filter((_, itemIndex) => itemIndex !== index))
          }
        />
      )}
      <BackgroundCredit />
    </div>
  );
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
