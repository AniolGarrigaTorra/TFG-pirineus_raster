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

export interface AoiBounds {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
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
  value_semantics?: string;
  native_resolution_m?: number;
  category_classes?: CategoryClassCatalog[];
  index?: number;
  resampling?: string;
  dataset?: string;
  layer?: string;
  geometry_type?: string;
  temporal?: Dict;
  generated_from?: string;
}

export interface CategoryClassCatalog {
  value?: string | number;
  values?: Array<string | number>;
  name?: string;
  label?: string;
  description?: string;
}

export interface TemporalCapability {
  kind: string;
  label?: string;
  temporal_axis?: string | null;
  aggregation_stage?: string;
  default_output_mode: string;
  output_modes: string[];
  aggregation_forms: string[];
  supports_custom_aggregations: boolean;
  supports_raw_slices: boolean;
  raw_timesteps_implemented?: boolean;
  default_months?: [number, number];
  default_years?: [number, number];
  available_years?: [number, number];
  postprocess_metrics?: string[];
  temporal_layers?: {
    annual?: boolean;
    annual_index?: boolean;
    months?: string[];
    seasons?: string[];
    years?: number[];
  };
  postprocess_outputs?: Dict[];
  dimensioned_by?: string[];
}

export interface SourceCatalog {
  id: string;
  title?: string;
  provider: string;
  provider_title?: string;
  provider_url?: string;
  product: string;
  product_group?: string;
  version?: string;
  description?: string;
  summary?: string;
  long_description?: string;
  official_url?: string;
  documentation_url?: string;
  page_url?: string;
  doi?: string;
  article_url?: string;
  citation?: string;
  config_path: string;
  source_crs?: string;
  source_period?: string;
  native_resolution?: string | number;
  native_resolution_m?: number;
  native_resolution_unit?: string;
  source_resolution?: string;
  source_resolution_options?: string[];
  target_resolution_m?: number;
  keep_raw_after_clip_default?: boolean;
  layer_structure?: string;
  file_format?: string;
  data_type?: string;
  variables?: VariableCatalog[];
  layers?: VariableCatalog[];
  dimensions?: Record<string, string[]>;
  aggregations?: Dict[];
  temporal?: TemporalCapability;
  resampling?: Dict;
}

export interface SourceGroupCatalog {
  id: string;
  title: string;
  official_url?: string;
  summary?: string;
  long_description?: string;
  references?: Array<{
    label: string;
    url: string;
  }>;
}

export interface WorkbenchCatalog {
  project: ProjectCatalog;
  aois: AoiCatalog[];
  source_groups?: SourceGroupCatalog[];
  sources: SourceCatalog[];
  supported_metrics: string[];
  supported_resampling: string[];
  derived_operation_groups?: Record<string, string[]>;
  value_semantics?: string[];
  supported_stages: string[];
}

export interface CustomAggregation {
  name: string;
  form: string;
  metric: string;
  months?: [number, number];
  years?: [number, number];
  within_year_metric?: string;
  across_year_metric?: string;
  output_metric_name?: string;
  threshold?: number;
  comparison?: string;
  start_date?: string;
  end_date?: string;
  variables: string[];
}

export interface CategoryFractionSelection {
  variable: string;
  name: string;
  class_values: Array<string | number>;
  label?: string;
}

export interface TemporalSelection {
  outputMode: string;
  months: [number, number];
  years?: [number, number];
  layers: {
    annual: boolean;
    annual_index: boolean;
    months: string[];
    seasons: string[];
    years: number[];
  };
  aggregationUse: string[];
  customAggregations: CustomAggregation[];
}

export interface SourceSelection {
  id: string;
  config: string;
  selected: boolean;
  stages: string[];
  sourceResolution?: string;
  keepRawAfterClip: boolean;
  variables: string[];
  layers: string[];
  categoryFractions: CategoryFractionSelection[];
  dimensions: Record<string, string[]>;
  temporal: TemporalSelection;
  resamplingByVariable: Record<string, string>;
}

export interface ThermalRangeRow {
  source_id: string;
  aggregation: string;
  gcm?: string;
  ssp?: string;
  period?: string;
}

export interface DerivedInputQuery {
  source_id?: string;
  provider?: string;
  product?: string;
  variable?: string;
  aggregation_name?: string;
  aggregation_metric?: string;
  months?: number[];
  gcm?: string;
  ssp?: string;
  period?: string;
}

export interface DerivedFeatureConfig {
  name: string;
  operation: "expression" | "recipe" | "terrain" | "focal" | "distance";
  recipe?: string;
  method?: string;
  expression?: string;
  description?: string;
  unit?: string;
  value_semantics?: string;
  valid_range?: number[];
  output_dtype?: string;
  temporal_meaning?: string;
  parameters?: Dict;
  inputs: Record<string, DerivedInputQuery>;
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
    temporal_output_mode?: string;
  }>;
}
