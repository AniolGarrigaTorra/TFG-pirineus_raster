export type Dict<T = unknown> = Record<string, T>;

export interface ProjectCatalog {
  config_path: string;
  name?: string;
  crs?: string;
  nodata?: number;
  available_resolutions_m: number[];
  default_resolution_m?: number;
}

export interface AoiCatalog {
  name: string;
  path: string;
  description?: string;
  crs?: string;
  bounds?: Dict<number>;
}

export interface VariableCatalog {
  name: string;
  kind: "variable" | "index" | "vector_layer";
  enabled_default?: boolean;
  description?: string;
  unit?: string | null;
  scale_factor?: number;
  valid_range?: number[];
  data_type?: string;
  native_resolution_m?: number;
  index?: number;
  dataset?: string;
  layer?: string;
  geometry_type?: string;
}

export interface SourceCatalog {
  id: string;
  provider: string;
  product: string;
  product_group?: string;
  version?: string;
  description?: string;
  config_path: string;
  source_crs?: string;
  source_period?: string;
  native_resolution?: string | number;
  native_resolution_m?: number;
  native_resolution_unit?: string;
  source_resolution?: string;
  target_resolution_m?: number;
  layer_structure?: string;
  file_format?: string;
  data_type?: string;
  variables?: VariableCatalog[];
  layers?: VariableCatalog[];
  dimensions?: Record<string, string[]>;
  aggregations?: Dict[];
  resampling?: Dict;
}

export interface WorkbenchCatalog {
  project: ProjectCatalog;
  aois: AoiCatalog[];
  sources: SourceCatalog[];
  supported_metrics: string[];
  supported_resampling: string[];
  supported_stages: string[];
}

export interface CustomAggregation {
  name: string;
  metric: string;
  months: [number, number];
  variables: string[];
}

export interface SourceSelection {
  id: string;
  config: string;
  selected: boolean;
  stages: string[];
  variables: string[];
  layers: string[];
  dimensions: Record<string, string[]>;
  aggregationUse: string[];
  customAggregations: CustomAggregation[];
  resamplingByVariable: Record<string, string>;
}

export interface ThermalRangeRow {
  source_id: string;
  aggregation: string;
  gcm?: string;
  ssp?: string;
  period?: string;
}

export interface ValidationReport {
  ok: boolean;
  errors: string[];
  warnings: string[];
  estimated_layers: number;
  estimated_source_layers?: number;
  estimated_derived_layers?: number;
  sources: Array<{
    id: string;
    provider?: string;
    product?: string;
    estimated_layers: number;
    variables?: string[];
    indices?: string[];
    aggregations?: string[];
  }>;
}
