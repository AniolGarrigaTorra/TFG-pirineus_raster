import { useEffect, useMemo, useState } from "react";
import { fetchCatalog, renderRunConfig, validateRunConfig } from "./api";
import type {
  AoiCatalog,
  CustomAggregation,
  SourceCatalog,
  SourceSelection,
  ThermalRangeRow,
  TemporalSelection,
  ValidationReport,
  WorkbenchCatalog
} from "./types";
import { renderYaml } from "./yaml";
import "./App.css";

const tabs = ["Project", "Sources", "Variables", "Temporal", "Derived", "Review"];

function sourceVariables(source: SourceCatalog) {
  return [...(source.variables ?? []), ...(source.layers ?? [])];
}

function sourceAggregationNames(source: SourceCatalog) {
  return (source.aggregations ?? [])
    .map((item) => String(item.name ?? ""))
    .filter(Boolean);
}

function defaultRange(value?: [number, number], fallback: [number, number] = [1, 12]): [number, number] {
  return Array.isArray(value) && value.length === 2 ? [Number(value[0]), Number(value[1])] : fallback;
}

function sourceTemporalSelection(source: SourceCatalog): TemporalSelection {
  const temporal = source.temporal;
  const layerOptions = temporal?.temporal_layers;

  return {
    outputMode: temporal?.default_output_mode ?? "static",
    months: defaultRange(temporal?.default_months),
    years: temporal?.default_years ? defaultRange(temporal.default_years) : undefined,
    layers: {
      annual: layerOptions?.annual ?? true,
      annual_index: layerOptions?.annual_index ?? true,
      months: [...(layerOptions?.months ?? [])],
      seasons: [...(layerOptions?.seasons ?? [])]
    },
    aggregationUse: sourceAggregationNames(source),
    customAggregations: []
  };
}

function toggleValue(values: string[], value: string) {
  return values.includes(value)
    ? values.filter((item) => item !== value)
    : [...values, value];
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
        months: item.months,
        variables: item.variables
      };
      if (item.years) base.years = item.years;
      if (item.form === "year_then_across_years") {
        base.within_year_metric = item.within_year_metric ?? "sum";
        base.across_year_metric = item.across_year_metric ?? "mean";
        if (item.output_metric_name) base.output_metric_name = item.output_metric_name;
      } else {
        base.metric = item.metric;
      }
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
        seasons: temporal.layers.seasons
      }
    };
  }

  if (mode === "postprocess_aggregate") {
    return {
      output_mode: "postprocess_aggregate"
    };
  }

  return undefined;
}

function createSelection(source: SourceCatalog): SourceSelection {
  const variables = sourceVariables(source)
    .filter((item) => item.enabled_default)
    .map((item) => item.name);

  const dimensions: Record<string, string[]> = {};
  for (const [key, values] of Object.entries(source.dimensions ?? {})) {
    dimensions[key] = [...values];
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
  const [aoiPath, setAoiPath] = useState("configs/aoi/experimental_pallars_sobira.yaml");
  const [resolution, setResolution] = useState(100);
  const [stages, setStages] = useState<string[]>(["build"]);
  const [datasetDir, setDatasetDir] = useState("data_processed/datasets/pallars_workbench_100m");
  const [selections, setSelections] = useState<Record<string, SourceSelection>>({});
  const [thermalRows, setThermalRows] = useState<ThermalRangeRow[]>([]);
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [serverYaml, setServerYaml] = useState("");
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    fetchCatalog()
      .then((data) => {
        setCatalog(data);
        setCatalogError(null);
        setProjectConfig(data.project.config_path);
        setResolution(data.project.default_resolution_m ?? data.project.available_resolutions_m[0] ?? 100);
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

  const selectedSources = useMemo(
    () => Object.values(selections).filter((selection) => selection.selected),
    [selections]
  );

  const selectedCatalogSources = useMemo(
    () => catalog?.sources.filter((source) => selections[source.id]?.selected) ?? [],
    [catalog?.sources, selections]
  );

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
        Object.entries(selection.dimensions).filter(([, values]) => values.length > 0)
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
        variables: variableNames.length > 0 ? variableNames : undefined,
        layers: layerNames.length > 0 ? layerNames : undefined,
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
        aoi_config: aoiPath,
        resolution_m: resolution,
        stages
      },
      sources: sourceEntries,
      derived_feature_groups: thermalRows.length > 0
        ? [
            {
              recipe: "thermal_range",
              foreach: thermalRows
            }
          ]
        : undefined,
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
    resolution,
    runName,
    selectedSources,
    stages,
    thermalRows
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

  function saveYaml() {
    const blob = new Blob([yamlText], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${runName}.yaml`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Pirineus Raster Workbench</h1>
          <p>{catalog?.project.crs ?? "EPSG:3035"} · {selectedSources.length} sources · {validation?.estimated_layers ?? 0} estimated layers</p>
        </div>
        <div className={`api-pill ${catalogError ? "bad" : catalog ? "good" : "loading"}`}>
          {apiStatus}
        </div>
      </header>

      <nav className="tabs" aria-label="Workbench sections">
        {tabs.map((tab) => (
          <button
            key={tab}
            className={activeTab === tab ? "active" : ""}
            onClick={() => setActiveTab(tab)}
          >
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
          runName={runName}
          setRunName={setRunName}
          description={description}
          setDescription={setDescription}
          projectConfig={projectConfig}
          setProjectConfig={setProjectConfig}
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
          thermalRows={thermalRows}
          setThermalRows={setThermalRows}
        />
      )}

      {activeTab === "Review" && (
        <ReviewPanel
          yamlText={yamlText}
          validation={validation}
          apiError={apiError}
          validate={validate}
          renderFromServer={renderFromServer}
          copyYaml={copyYaml}
          saveYaml={saveYaml}
        />
      )}
    </div>
  );
}

interface ProjectPanelProps {
  catalog: WorkbenchCatalog | null;
  runName: string;
  setRunName: (value: string) => void;
  description: string;
  setDescription: (value: string) => void;
  projectConfig: string;
  setProjectConfig: (value: string) => void;
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
  const aois = props.catalog?.aois ?? [];
  const resolutions = props.catalog?.project.available_resolutions_m ?? [100];
  const supportedStages = props.catalog?.supported_stages ?? ["build"];

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
            AOI
            <select value={props.aoiPath} onChange={(event) => props.setAoiPath(event.target.value)}>
              {aois.map((aoi: AoiCatalog) => (
                <option key={aoi.path} value={aoi.path}>{aoi.name}</option>
              ))}
            </select>
          </label>
          <label>
            Resolution
            <select value={props.resolution} onChange={(event) => props.setResolution(Number(event.target.value))}>
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
                onChange={() => props.setStages(toggleValue(props.stages, stage))}
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
  return (
    <main className="workspace">
      <section className="source-grid">
        {catalog.sources.map((source) => {
          const selection = selections[source.id];
          const sourceResolutionOptions = source.source_resolution_options?.length
            ? source.source_resolution_options
            : source.source_resolution
              ? [source.source_resolution]
              : [];
          const sourceResolution = selection?.sourceResolution ?? source.source_resolution ?? "";
          return (
            <article key={source.id} className={`source-card ${selection?.selected ? "selected" : ""}`}>
              <div className="source-head">
                <label className="switch-row">
                  <input
                    type="checkbox"
                    checked={selection?.selected ?? false}
                    onChange={(event) => patchSelection(source.id, { selected: event.target.checked })}
                  />
                  <span>{source.id}</span>
                </label>
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
              <div className="source-meta">
                <span>{source.provider}</span>
                <span>{source.product}</span>
                <span>{source.native_resolution ?? source.source_resolution ?? "native"}</span>
              </div>
              <p>{source.description ?? source.layer_structure}</p>
              {selection?.selected && (
                <div className="source-controls">
                  <label className="switch-row subtle">
                    <input
                      type="checkbox"
                      checked={selection.stages.length === 0}
                      onChange={(event) =>
                        patchSelection(source.id, {
                          stages: event.target.checked ? [] : ["build"]
                        })
                      }
                    />
                    <span>Use project stages</span>
                  </label>
                  {selection.stages.length > 0 && (
                    <div className="choice-list compact">
                      {catalog.supported_stages.map((stage) => (
                        <label key={stage} className="check-row">
                          <input
                            type="checkbox"
                            checked={selection.stages.includes(stage)}
                            onChange={() =>
                              patchSelection(source.id, {
                                stages: toggleValue(selection.stages, stage)
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
          <option key={source.id} value={source.id}>{source.id}</option>
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

  return (
    <main className="workspace two-col wide-left">
      <section className="panel">
        <div className="panel-head">
          <h2>Variables</h2>
          <SourceChooser sources={sources} activeSourceId={activeSourceId} setActiveSourceId={setActiveSourceId} />
        </div>
        <div className="choice-list">
          {variables.map((variable) => (
            <label key={variable.name} className="check-row rich">
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
                <strong>{variable.name}</strong>
                <small>{variable.description ?? variable.kind}</small>
              </span>
              <em>{variable.unit ?? variable.geometry_type ?? ""}</em>
            </label>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Dimensions</h2>
        {Object.entries(source.dimensions ?? {}).map(([key, values]) => (
          <div className="dimension-block" key={key}>
            <h3>{key}</h3>
            <div className="choice-list compact">
              {values.map((value) => (
                <label key={value} className="check-row">
                  <input
                    type="checkbox"
                    checked={(selection.dimensions[key] ?? []).includes(value)}
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
        ))}
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
  const defaultForm = isTimeSeries ? "year_then_across_years" : "month_range_metric";
  const [custom, setCustom] = useState<CustomAggregation>({
    name: "custom_mean",
    form: defaultForm,
    metric: "mean",
    months: [1, 12],
    years: temporal.years,
    within_year_metric: "sum",
    across_year_metric: "mean",
    variables: selection.variables.slice(0, 1)
  });

  useEffect(() => {
    setCustom({
      name: isTimeSeries ? "custom_period" : "custom_mean",
      form: isTimeSeries ? "year_then_across_years" : "month_range_metric",
      metric: "mean",
      months: temporal.months,
      years: temporal.years,
      within_year_metric: "sum",
      across_year_metric: "mean",
      output_metric_name: isTimeSeries ? "mean_period_sum" : undefined,
      variables: selection.variables.slice(0, 1)
    });
  }, [source.id]);

  const selectedCustomVariables = custom.variables.filter((variable) => selection.variables.includes(variable));
  const canAddCustom = custom.name.trim().length > 0 && selectedCustomVariables.length > 0;
  const supportsAggregate = capability?.output_modes.includes("aggregate") ?? false;
  const supportsRaw = capability?.output_modes.includes("raw_slices") ?? false;
  const supportsSupplied = capability?.output_modes.includes("supplied_layers") ?? false;
  const supportsPostprocess = capability?.output_modes.includes("postprocess_aggregate") ?? false;

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
            <div className="choice-list compact">
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={temporal.layers.annual}
                  onChange={() => patchTemporalLayers({ annual: !temporal.layers.annual })}
                />
                <span>Annual layers</span>
              </label>
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={temporal.layers.annual_index}
                  onChange={() => patchTemporalLayers({ annual_index: !temporal.layers.annual_index })}
                />
                <span>Annual index layers</span>
              </label>
            </div>
            <h3>Months</h3>
            <div className="choice-list compact token-grid">
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
            <h3>Seasons</h3>
            <div className="choice-list compact token-grid">
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
          </div>
        )}

        {temporal.outputMode === "postprocess_aggregate" && supportsPostprocess && (
          <>
            <div className="notice info">
              This source creates temporal products during download/postprocess. Select the generated output variables in the Variables tab.
            </div>
            <div className="aggregation-list">
              {(capability?.postprocess_outputs ?? []).map((item) => (
                <div className="aggregation-chip" key={String(item.name)}>
                  <span>
                    <strong>{String(item.name)}</strong>
                    <small>
                      {String(item.method ?? "")}
                      {Array.isArray(item.months) ? ` · months ${item.months.join(", ")}` : ""}
                    </small>
                  </span>
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
              {isTimeSeries && (
                <>
                  <label>
                    Start year
                    <input type="number" value={custom.years?.[0] ?? ""} onChange={(event) => setCustom({ ...custom, years: [Number(event.target.value), custom.years?.[1] ?? Number(event.target.value)] })} />
                  </label>
                  <label>
                    End year
                    <input type="number" value={custom.years?.[1] ?? ""} onChange={(event) => setCustom({ ...custom, years: [custom.years?.[0] ?? Number(event.target.value), Number(event.target.value)] })} />
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
              <label>
                Start month
                <input type="number" min={1} max={12} value={custom.months[0]} onChange={(event) => setCustom({ ...custom, months: [Number(event.target.value), custom.months[1]] })} />
              </label>
              <label>
                End month
                <input type="number" min={1} max={12} value={custom.months[1]} onChange={(event) => setCustom({ ...custom, months: [custom.months[0], Number(event.target.value)] })} />
              </label>
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
                      · months {item.months.join("-")} · {item.variables.join(", ")}
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

function DerivedPanel({
  selectedSources,
  catalog,
  thermalRows,
  setThermalRows
}: {
  selectedSources: SourceSelection[];
  catalog: WorkbenchCatalog;
  thermalRows: ThermalRangeRow[];
  setThermalRows: (rows: ThermalRangeRow[]) => void;
}) {
  const eligibleSources = selectedSources.filter((selection) => {
    const names = new Set(selection.variables);
    const aggregationNames = [
      ...selection.temporal.aggregationUse,
      ...selection.temporal.customAggregations.map((item) => item.name)
    ];
    return names.has("tmin") && names.has("tmax") && aggregationNames.length > 0;
  });
  const first = eligibleSources[0];
  const firstCatalog = catalog.sources.find((source) => source.id === first?.id);

  function aggregationOptions(selection?: SourceSelection) {
    if (!selection) return [];
    return [
      ...selection.temporal.aggregationUse,
      ...selection.temporal.customAggregations.map((item) => item.name)
    ];
  }

  function addThermalRange() {
    if (!first || !firstCatalog) return;
    const aggregations = aggregationOptions(first);
    setThermalRows([
      ...thermalRows,
      {
        source_id: first.id,
        aggregation: aggregations[0] ?? sourceAggregationNames(firstCatalog)[0] ?? "annual_mean",
        gcm: first.dimensions.gcms?.[0],
        ssp: first.dimensions.ssps?.[0],
        period: first.dimensions.periods?.[0]
      }
    ]);
  }

  return (
    <main className="workspace">
      <section className="panel">
        <div className="panel-head">
          <h2>Derived Features</h2>
          <button className="primary" onClick={addThermalRange} disabled={!first}>Add thermal range</button>
        </div>
        {selectedSources.length === 0 && (
          <div className="notice info">
            Select at least one source before adding derived features.
          </div>
        )}
        {selectedSources.length > 0 && eligibleSources.length === 0 && (
          <div className="notice info">
            Thermal range needs a selected source with tmin, tmax and at least one aggregation.
          </div>
        )}
        <div className="table">
          <div className="table-row table-head">
            <span>Source</span>
            <span>Aggregation</span>
            <span>GCM</span>
            <span>SSP</span>
            <span>Period</span>
            <span></span>
          </div>
          {thermalRows.map((row, index) => {
            const sourceSelection = eligibleSources.find((item) => item.id === row.source_id);
            return (
              <div className="table-row" key={`${row.source_id}-${index}`}>
                <select value={row.source_id} onChange={(event) => {
                  const source_id = event.target.value;
                  const nextSource = eligibleSources.find((item) => item.id === source_id);
                  const next = [...thermalRows];
                  next[index] = {
                    ...row,
                    source_id,
                    aggregation: aggregationOptions(nextSource)[0] ?? row.aggregation,
                    gcm: nextSource?.dimensions.gcms?.[0],
                    ssp: nextSource?.dimensions.ssps?.[0],
                    period: nextSource?.dimensions.periods?.[0]
                  };
                  setThermalRows(next);
                }}>
                  {eligibleSources.map((item) => (
                    <option key={item.id} value={item.id}>{item.id}</option>
                  ))}
                </select>
                <select value={row.aggregation} onChange={(event) => {
                  const next = [...thermalRows];
                  next[index] = { ...row, aggregation: event.target.value };
                  setThermalRows(next);
                }}>
                  {aggregationOptions(sourceSelection).map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
                <input value={row.gcm ?? ""} onChange={(event) => {
                  const next = [...thermalRows];
                  next[index] = { ...row, gcm: event.target.value || undefined };
                  setThermalRows(next);
                }} />
                <input value={row.ssp ?? ""} onChange={(event) => {
                  const next = [...thermalRows];
                  next[index] = { ...row, ssp: event.target.value || undefined };
                  setThermalRows(next);
                }} />
                <input value={row.period ?? ""} onChange={(event) => {
                  const next = [...thermalRows];
                  next[index] = { ...row, period: event.target.value || undefined };
                  setThermalRows(next);
                }} />
                <button className="ghost danger" onClick={() => setThermalRows(thermalRows.filter((_, rowIndex) => rowIndex !== index))}>Remove</button>
              </div>
            );
          })}
        </div>
      </section>
    </main>
  );
}

function ReviewPanel({
  yamlText,
  validation,
  apiError,
  validate,
  renderFromServer,
  copyYaml,
  saveYaml
}: {
  yamlText: string;
  validation: ValidationReport | null;
  apiError: string | null;
  validate: () => void;
  renderFromServer: () => void;
  copyYaml: () => void;
  saveYaml: () => void;
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
            <button onClick={saveYaml}>Save YAML</button>
          </div>
        </div>
        {apiError && <div className="notice error">{apiError}</div>}
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

      <section className="panel yaml-panel">
        <h2>YAML</h2>
        <pre>{yamlText}</pre>
      </section>
    </main>
  );
}

export default App;
