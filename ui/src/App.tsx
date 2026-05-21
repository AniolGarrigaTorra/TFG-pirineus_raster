import { useEffect, useMemo, useState } from "react";
import { fetchCatalog, renderRunConfig, validateRunConfig } from "./api";
import type {
  AoiCatalog,
  CustomAggregation,
  DerivedFeatureConfig,
  DerivedInputQuery,
  SourceCatalog,
  SourceSelection,
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
  const [targetCrs, setTargetCrs] = useState("EPSG:3035");
  const [aoiPath, setAoiPath] = useState("configs/aoi/experimental_pallars_sobira.yaml");
  const [resolution, setResolution] = useState(100);
  const [stages, setStages] = useState<string[]>(["build"]);
  const [datasetDir, setDatasetDir] = useState("data_processed/datasets/pallars_workbench_100m");
  const [selections, setSelections] = useState<Record<string, SourceSelection>>({});
  const [derivedFeatures, setDerivedFeatures] = useState<DerivedFeatureConfig[]>([]);
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [serverYaml, setServerYaml] = useState("");
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    fetchCatalog()
      .then((data) => {
        setCatalog(data);
        setCatalogError(null);
        setProjectConfig(data.project.config_path);
        setTargetCrs(data.project.crs ?? "EPSG:3035");
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
          <p>{targetCrs} · {selectedSources.length} sources · {validation?.estimated_layers ?? 0} estimated layers</p>
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
          <details key={group.provider} className="source-group" open>
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
                                stages: event.target.checked ? [] : ["build"]
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
          <option key={source.id} value={source.id}>{sourceDisplayName(source)} ({source.id})</option>
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
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Dimensions</h2>
        {Object.entries(source.dimensions ?? {}).length === 0 && (
          <div className="notice info">This source has no selectable dimensions.</div>
        )}
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

        <div className="dimension-block">
          <h2>Resampling</h2>
          <div className="choice-list compact">
            {selectedVariableItems.map((variable) => {
              const defaultMethod = variable.resampling ?? "nearest";
              const currentMethod = selection.resamplingByVariable[variable.name] ?? defaultMethod;
              return (
                <label key={variable.name}>
                  {variable.name}
                  <select
                    value={currentMethod}
                    onChange={(event) =>
                      patchSelectionMut(source.id, (current) => {
                        const next = { ...current.resamplingByVariable };
                        if (event.target.value === defaultMethod) {
                          delete next[variable.name];
                        } else {
                          next[variable.name] = event.target.value;
                        }
                        return { ...current, resamplingByVariable: next };
                      })
                    }
                  >
                    {catalog.supported_resampling.map((method) => (
                      <option key={method} value={method}>
                        {method}{method === defaultMethod ? " (default)" : ""}
                      </option>
                    ))}
                  </select>
                  <small className="field-hint">
                    {variable.value_semantics ?? variable.data_type ?? "continuous"}
                  </small>
                </label>
              );
            })}
          </div>
          {(catalog.advanced_interpolation_methods ?? []).length > 0 && (
            <div className="advanced-methods">
              <strong>Advanced interpolation backends</strong>
              {(catalog.advanced_interpolation_methods ?? []).map((method) => (
                <span key={String(method.name)}>{String(method.label ?? method.name)} · {String(method.status)}</span>
              ))}
            </div>
          )}
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

interface PlannedLayer {
  id: string;
  label: string;
  sourceTitle: string;
  query: DerivedInputQuery;
  variable: string;
  unit?: string | null;
  valueSemantics?: string;
}

function buildPlannedLayers(
  catalog: WorkbenchCatalog,
  selectedSources: SourceSelection[]
): PlannedLayer[] {
  const layers: PlannedLayer[] = [];

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
    const gcms = selection.dimensions.gcms?.length ? selection.dimensions.gcms : [undefined];
    const ssps = selection.dimensions.ssps?.length ? selection.dimensions.ssps : [undefined];
    const periods = selection.dimensions.periods?.length ? selection.dimensions.periods : [undefined];

    for (const variable of variables) {
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
                sourceDisplayName(source),
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
  const [recipe, setRecipe] = useState("thermal_range");
  const [primaryLayerId, setPrimaryLayerId] = useState("");
  const [secondaryLayerId, setSecondaryLayerId] = useState("");
  const [outputName, setOutputName] = useState("");
  const [expression, setExpression] = useState("x - y");
  const [unit, setUnit] = useState("");
  const [valueSemantics, setValueSemantics] = useState("intensive");
  const [description, setDescription] = useState("");
  const [method, setMethod] = useState("slope");
  const [radius, setRadius] = useState(1);
  const [threshold, setThreshold] = useState(0);
  const [classValue, setClassValue] = useState(1);

  const layerById = useMemo(
    () => new Map(plannedLayers.map((layer) => [layer.id, layer])),
    [plannedLayers]
  );
  const firstLayer = plannedLayers[0];
  const primaryLayer = layerById.get(primaryLayerId) ?? firstLayer;
  const secondaryLayer = layerById.get(secondaryLayerId) ?? plannedLayers[1] ?? firstLayer;

  function findLayer(variable: string, sameAs?: PlannedLayer) {
    return plannedLayers.find((layer) =>
      layer.variable === variable &&
      (!sameAs ||
        (
          layer.query.source_id === sameAs.query.source_id &&
          layer.query.aggregation_name === sameAs.query.aggregation_name &&
          layer.query.gcm === sameAs.query.gcm &&
          layer.query.ssp === sameAs.query.ssp &&
          layer.query.period === sameAs.query.period
        ))
    );
  }

  function addFeature(feature: DerivedFeatureConfig) {
    setDerivedFeatures([...derivedFeatures, feature]);
    setOutputName("");
    setDescription("");
  }

  function addGuidedRecipe() {
    if (!primaryLayer) return;

    let inputs: Record<string, DerivedInputQuery> = {};
    let parameters: Record<string, unknown> = {};
    let defaultUnit = unit || primaryLayer.unit || "unitless";
    let defaultExpressionName = recipe;

    if (recipe === "thermal_range") {
      const tmax = findLayer("tmax", primaryLayer) ?? findLayer("tmax");
      const tmin = findLayer("tmin", tmax ?? primaryLayer) ?? findLayer("tmin");
      if (!tmax || !tmin) return;
      inputs = { tmax: tmax.query, tmin: tmin.query };
      defaultUnit = unit || "degC";
    } else if (recipe === "water_balance" || recipe === "aridity_index") {
      const prec = findLayer("prec", primaryLayer) ?? findLayer("prec");
      const pet = findLayer("pet", prec ?? primaryLayer) ?? findLayer("pet");
      if (!prec || !pet) return;
      inputs = { prec: prec.query, pet: pet.query };
      defaultUnit = recipe === "aridity_index" ? "ratio" : (unit || "mm");
      parameters = recipe === "aridity_index" ? { convention: "prec_over_pet" } : {};
    } else if (recipe === "snow_persistence_ratio") {
      const snow = findLayer("snow_days", primaryLayer) ?? primaryLayer;
      const valid = findLayer("valid_days", snow) ?? secondaryLayer;
      if (!snow || !valid) return;
      inputs = { snow_days: snow.query, valid_days: valid.query };
      defaultUnit = "ratio";
    } else if (recipe === "seasonal_contrast") {
      if (!secondaryLayer) return;
      inputs = { a: primaryLayer.query, b: secondaryLayer.query };
      parameters = { metric: "difference" };
      defaultUnit = unit || primaryLayer.unit || "source_units";
    } else if (recipe === "binary_threshold_mask") {
      inputs = { x: primaryLayer.query };
      parameters = { operator: ">=", threshold };
      defaultUnit = "binary";
      defaultExpressionName = `${primaryLayer.variable}_threshold_mask`;
    } else if (recipe === "class_mask") {
      inputs = { x: primaryLayer.query };
      parameters = { class_value: classValue };
      defaultUnit = "binary";
      defaultExpressionName = `${primaryLayer.variable}_class_${classValue}_mask`;
    }

    addFeature({
      name: sanitizeDerivedName(outputName || defaultExpressionName),
      operation: "recipe",
      recipe,
      description: description || humanizeId(outputName || defaultExpressionName),
      unit: defaultUnit,
      value_semantics: recipe.includes("mask") ? "categorical" : valueSemantics,
      output_dtype: recipe.includes("mask") ? "uint8" : "float32",
      parameters,
      inputs
    });
  }

  function addExpression() {
    if (!primaryLayer) return;
    const inputs: Record<string, DerivedInputQuery> = { x: primaryLayer.query };
    if (secondaryLayer) inputs.y = secondaryLayer.query;
    addFeature({
      name: sanitizeDerivedName(outputName || "custom_expression"),
      operation: "expression",
      expression,
      description: description || "Custom derived raster expression.",
      unit: unit || primaryLayer.unit || "unitless",
      value_semantics: valueSemantics,
      output_dtype: "float32",
      inputs
    });
  }

  function addSpatialOperation(operation: "terrain" | "focal" | "distance") {
    if (!primaryLayer) return;
    const spatialMethod = operation === "terrain" ? method : operation === "focal" ? method : "distance_to_mask";
    addFeature({
      name: sanitizeDerivedName(outputName || `${primaryLayer.variable}_${spatialMethod}`),
      operation,
      method: spatialMethod,
      description: description || `${humanizeId(spatialMethod)} derived from ${primaryLayer.label}.`,
      unit: operation === "distance" ? "m" : unit || primaryLayer.unit || "unitless",
      value_semantics: operation === "distance" ? "intensive" : valueSemantics,
      output_dtype: "float32",
      parameters: {
        radius: operation === "focal" || ["ruggedness", "tpi", "roughness"].includes(spatialMethod) ? radius : undefined,
        class_value: operation === "distance" ? classValue : undefined
      },
      inputs: {
        [operation === "terrain" ? "dem" : operation === "distance" ? "mask" : "x"]: primaryLayer.query
      }
    });
  }

  return (
    <main className="workspace two-col wide-left">
      <section className="panel">
        <div className="panel-head">
          <h2>Derived Features</h2>
          <span className="field-hint">{plannedLayers.length} planned input layers</span>
        </div>
        {selectedSources.length === 0 && (
          <div className="notice info">Select at least one source before adding derived features.</div>
        )}
        {selectedSources.length > 0 && plannedLayers.length === 0 && (
          <div className="notice info">Select variables before adding derived features.</div>
        )}

        <div className="derived-builder">
          <h3>Guided recipes</h3>
          <div className="form-grid">
            <label>
              Recipe
              <select value={recipe} onChange={(event) => setRecipe(event.target.value)}>
                <option value="thermal_range">Thermal range</option>
                <option value="water_balance">Water balance</option>
                <option value="aridity_index">Aridity index</option>
                <option value="seasonal_contrast">Seasonal contrast</option>
                <option value="snow_persistence_ratio">Snow persistence ratio</option>
                <option value="binary_threshold_mask">Binary threshold mask</option>
                <option value="class_mask">Class mask</option>
              </select>
            </label>
            <label>
              Main input
              <select value={primaryLayer?.id ?? ""} onChange={(event) => setPrimaryLayerId(event.target.value)}>
                {plannedLayers.map((layer) => (
                  <option key={layer.id} value={layer.id}>{layer.label}</option>
                ))}
              </select>
            </label>
            <label>
              Secondary input
              <select value={secondaryLayer?.id ?? ""} onChange={(event) => setSecondaryLayerId(event.target.value)}>
                {plannedLayers.map((layer) => (
                  <option key={layer.id} value={layer.id}>{layer.label}</option>
                ))}
              </select>
            </label>
            <label>
              Output name
              <input value={outputName} onChange={(event) => setOutputName(event.target.value)} placeholder="auto if blank" />
            </label>
            <label>
              Threshold
              <input type="number" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
            </label>
            <label>
              Class value
              <input type="number" value={classValue} onChange={(event) => setClassValue(Number(event.target.value))} />
            </label>
          </div>
          <button className="primary" onClick={addGuidedRecipe} disabled={!primaryLayer}>Add guided feature</button>
        </div>

        <div className="derived-builder">
          <h3>Advanced expression</h3>
          <div className="form-grid">
            <label className="span-2">
              Expression
              <input value={expression} onChange={(event) => setExpression(event.target.value)} />
              <small className="field-hint">Use aliases x and y, plus safe functions such as where, sqrt, log, minimum and maximum.</small>
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
        </div>
      </section>

      <section className="panel">
        <h2>Spatial operations</h2>
        <div className="form-grid single">
          <label>
            Spatial method
            <select value={method} onChange={(event) => setMethod(event.target.value)}>
              <option value="slope">Slope</option>
              <option value="aspect">Aspect</option>
              <option value="ruggedness">Ruggedness</option>
              <option value="tpi">TPI</option>
              <option value="roughness">Roughness</option>
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
            Radius in cells
            <input type="number" min={1} value={radius} onChange={(event) => setRadius(Number(event.target.value))} />
          </label>
          <div className="button-row">
            <button onClick={() => addSpatialOperation("terrain")} disabled={!primaryLayer}>Add terrain</button>
            <button onClick={() => addSpatialOperation("focal")} disabled={!primaryLayer}>Add focal</button>
            <button onClick={() => addSpatialOperation("distance")} disabled={!primaryLayer}>Add distance-to</button>
          </div>
        </div>

        <div className="advanced-methods">
          <strong>Geostatistical interpolation</strong>
          <span>IDW, kriging and splines are intentionally kept outside derived features. They create rasters from points/covariates and belong in a future interpolation module.</span>
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
