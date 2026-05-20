import { useEffect, useMemo, useState } from "react";
import { fetchCatalog, renderRunConfig, validateRunConfig } from "./api";
import type {
  AoiCatalog,
  CustomAggregation,
  SourceCatalog,
  SourceSelection,
  ThermalRangeRow,
  ValidationReport,
  WorkbenchCatalog
} from "./types";
import { renderYaml } from "./yaml";
import "./App.css";

const tabs = ["Project", "Sources", "Variables", "Aggregations", "Derived", "Review"];

function sourceVariables(source: SourceCatalog) {
  return [...(source.variables ?? []), ...(source.layers ?? [])];
}

function sourceAggregationNames(source: SourceCatalog) {
  return (source.aggregations ?? [])
    .map((item) => String(item.name ?? ""))
    .filter(Boolean);
}

function toggleValue(values: string[], value: string) {
  return values.includes(value)
    ? values.filter((item) => item !== value)
    : [...values, value];
}

function createSelection(source: SourceCatalog): SourceSelection {
  const variables = sourceVariables(source)
    .filter((item) => item.enabled_default)
    .map((item) => item.name);

  const dimensions: Record<string, string[]> = {};
  for (const [key, values] of Object.entries(source.dimensions ?? {})) {
    dimensions[key] = values.length > 0 ? [values[0]] : [];
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
    aggregationUse: sourceAggregationNames(source),
    customAggregations: [],
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

      const custom = selection.customAggregations.map((item) => ({
        name: item.name,
        months: item.months,
        metric: item.metric,
        variables: item.variables
      }));
      const aggregations = selection.aggregationUse.length > 0 || custom.length > 0
        ? {
            use: selection.aggregationUse,
            custom
          }
        : undefined;
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
        aggregations
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

      {activeTab === "Aggregations" && catalog && activeSource && (
        <AggregationPanel
          catalog={catalog}
          sources={selectedCatalogSources}
          source={activeSource}
          selection={selections[activeSource.id]}
          activeSourceId={activeSourceId}
          setActiveSourceId={setActiveSourceId}
          patchSelectionMut={patchSelectionMut}
        />
      )}

      {activeTab === "Aggregations" && catalog && !activeSource && (
        <NoSelectedSourcesPanel title="Aggregations" />
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

function AggregationPanel({
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
  const [custom, setCustom] = useState<CustomAggregation>({
    name: "custom_mean",
    metric: "mean",
    months: [1, 12],
    variables: selection.variables.slice(0, 1)
  });

  const canAddCustom = custom.name.trim().length > 0 && custom.variables.length > 0;

  return (
    <main className="workspace two-col wide-left">
      <section className="panel custom-aggregation-panel">
        <div className="panel-head">
          <h2>Custom Aggregation</h2>
          <SourceChooser sources={sources} activeSourceId={activeSourceId} setActiveSourceId={setActiveSourceId} />
        </div>
        <div className="form-grid custom-aggregation-grid">
          <label>
            Name
            <input value={custom.name} onChange={(event) => setCustom({ ...custom, name: event.target.value })} />
          </label>
          <label>
            Metric
            <select value={custom.metric} onChange={(event) => setCustom({ ...custom, metric: event.target.value })}>
              {catalog.supported_metrics.map((metric) => (
                <option key={metric} value={metric}>{metric}</option>
              ))}
            </select>
          </label>
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
                name: custom.name.trim()
              };
              patchSelectionMut(source.id, (current) => ({
                ...current,
                customAggregations: [...current.customAggregations, nextCustom]
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
          {selection.customAggregations.map((item, index) => (
            <div className="aggregation-chip" key={`${item.name}-${index}`}>
              <span>
                <strong>{item.name}</strong>
                <small>{item.metric} · months {item.months.join("-")} · {item.variables.join(", ")}</small>
              </span>
              <button
                className="ghost danger"
                onClick={() =>
                  patchSelectionMut(source.id, (current) => ({
                    ...current,
                    customAggregations: current.customAggregations.filter((_, itemIndex) => itemIndex !== index)
                  }))
                }
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="panel preset-panel">
        <h2>Aggregation Presets</h2>
        <div className="choice-list compact">
          {(source.aggregations ?? []).map((aggregation) => {
            const name = String(aggregation.name);
            return (
              <label key={name} className="check-row rich preset-row">
                <input
                  type="checkbox"
                  checked={selection.aggregationUse.includes(name)}
                  onChange={() =>
                    patchSelectionMut(source.id, (current) => ({
                      ...current,
                      aggregationUse: toggleValue(current.aggregationUse, name)
                    }))
                  }
                />
                <span>
                  <strong>{name}</strong>
                  <small>{String(aggregation.metric ?? aggregation.output_metric_name ?? "two-step")}</small>
                </span>
                <em>{Array.isArray(aggregation.months) ? aggregation.months.join("-") : ""}</em>
              </label>
            );
          })}
        </div>
        {source.aggregations?.length === 0 && (
          <div className="notice info">
            This source does not define preset aggregations.
          </div>
        )}
        {selection.aggregationUse.length > 0 && (
          <div className="aggregation-list compact-summary">
            {selection.aggregationUse.map((name) => (
              <div className="aggregation-chip preset-chip" key={name}>
                <span>{name}</span>
                <button
                  className="ghost danger"
                  onClick={() =>
                    patchSelectionMut(source.id, (current) => ({
                      ...current,
                      aggregationUse: current.aggregationUse.filter((item) => item !== name)
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
      ...selection.aggregationUse,
      ...selection.customAggregations.map((item) => item.name)
    ];
    return names.has("tmin") && names.has("tmax") && aggregationNames.length > 0;
  });
  const first = eligibleSources[0];
  const firstCatalog = catalog.sources.find((source) => source.id === first?.id);

  function aggregationOptions(selection?: SourceSelection) {
    if (!selection) return [];
    return [
      ...selection.aggregationUse,
      ...selection.customAggregations.map((item) => item.name)
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
