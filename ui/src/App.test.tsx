import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { WorkbenchCatalog } from "./types";

const catalog: WorkbenchCatalog = {
  project: {
    config_path: "configs/project.yaml",
    crs: "EPSG:3035",
    available_resolutions_m: [100],
    default_resolution_m: 100
  },
  aois: [
    {
      name: "experimental_pallars_sobira",
      path: "configs/aoi/experimental_pallars_sobira.yaml"
    }
  ],
  sources: [
    {
      id: "worldclim_cmip6_future",
      provider: "worldclim",
      product: "cmip6_future",
      config_path: "configs/sources/worldclim/worldclim_cmip6_future.yaml",
      variables: [
        { name: "tmin", kind: "variable", enabled_default: true },
        { name: "tmax", kind: "variable", enabled_default: true }
      ],
      dimensions: {
        gcms: ["ACCESS-CM2"],
        ssps: ["ssp126"],
        periods: ["2021-2040"]
      },
      aggregations: [
        { name: "annual_mean", metric: "mean", months: [1, 12] }
      ],
      temporal: {
        kind: "future_monthly",
        label: "Future monthly climatology",
        default_output_mode: "aggregate",
        output_modes: ["aggregate", "raw_slices"],
        aggregation_forms: ["month_range_metric"],
        supports_custom_aggregations: true,
        supports_raw_slices: true,
        default_months: [1, 12],
        dimensioned_by: ["gcms", "ssps", "periods"]
      }
    }
  ],
  supported_metrics: ["mean", "sum"],
  supported_resampling: ["nearest", "average"],
  supported_stages: ["download", "clip", "build", "all"]
};

function mockApi(nextCatalog: WorkbenchCatalog) {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/backgrounds/manifest.json")) {
      return {
        ok: false,
        text: async () => "",
        json: async () => ({})
      } as Response;
    }
    if (url.includes("/api/validate-run")) {
      return {
        ok: true,
        text: async () => JSON.stringify({
          ok: true,
          errors: [],
          warnings: [],
          estimated_layers: 1,
          estimated_source_layers: 1,
          estimated_derived_layers: 0,
          sources: []
        })
      } as Response;
    }
    return {
      ok: true,
      text: async () => JSON.stringify(nextCatalog)
    } as Response;
  }));
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the workbench with catalog sources", async () => {
    mockApi(catalog);

    const { container } = render(<App />);

    expect(screen.getByText("Welcome to Pirineus Raster")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Start building my personalized dataset" }));

    expect(screen.getByText("Start new project")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Start new project/i }));

    expect(screen.getByText("Pirineus Raster Workbench")).toBeTruthy();

    await waitFor(() => {
      expect(screen.getByText("API ready")).toBeTruthy();
    });

    expect((screen.getByLabelText("all") as HTMLInputElement).checked).toBe(true);
    expect((screen.getByLabelText("build") as HTMLInputElement).checked).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Sources" }));

    const sourceGroup = container.querySelector(".source-group");
    expect(sourceGroup?.hasAttribute("open")).toBe(false);
    fireEvent.click(screen.getByText("Worldclim"));

    expect(screen.getByText("worldclim_cmip6_future")).toBeTruthy();
    expect((screen.getByRole("checkbox", { name: /worldclim_cmip6_future/i }) as HTMLInputElement).checked).toBe(false);
  });

  it("keeps AOI creation in the dedicated AOI workflow", async () => {
    mockApi(catalog);

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Start building my personalized dataset" }));
    fireEvent.click(screen.getByRole("button", { name: /Start new project/i }));

    await waitFor(() => {
      expect(screen.getByText("API ready")).toBeTruthy();
    });

    expect(screen.queryByRole("button", { name: "Create AOI config" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Home" }));
    fireEvent.click(screen.getByRole("button", { name: /New AOI/i }));

    expect(screen.getByRole("button", { name: "Create AOI config" })).toBeTruthy();
  });

  it("draws a projected AOI footprint for EPSG:3035 bounds", async () => {
    mockApi(catalog);

    const { container } = render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Start building my personalized dataset" }));
    fireEvent.click(screen.getByRole("button", { name: /New AOI/i }));

    fireEvent.change(screen.getByLabelText("xmin"), { target: { value: "3375000" } });
    fireEvent.change(screen.getByLabelText("xmax"), { target: { value: "3716000" } });
    fireEvent.change(screen.getByLabelText("ymin"), { target: { value: "2140000" } });
    fireEvent.change(screen.getByLabelText("ymax"), { target: { value: "2300000" } });

    const overlayPath = container.querySelector(".bbox-overlay .bbox-selection");
    expect(overlayPath?.getAttribute("d")).toContain("M");
    expect(overlayPath?.getAttribute("d")).toContain("L");
  });

  it("shows a single explicit input variable for snow postprocess aggregations", async () => {
    mockApi({
      ...catalog,
      sources: [
        {
          id: "copernicus_hrsi_snow",
          title: "Copernicus HRSI Fractional Snow Cover",
          provider: "copernicus",
          product: "hrsi_snow",
          config_path: "configs/sources/copernicus/copernicus_hrsi_snow.yaml",
          variables: [
            {
              name: "snow_fraction",
              kind: "variable",
              enabled_default: true,
              description: "Daily fractional snow cover on ground."
            }
          ],
          temporal: {
            kind: "temporal_postprocess",
            label: "Download-time temporal postprocess",
            default_output_mode: "postprocess_aggregate",
            output_modes: ["postprocess_aggregate"],
            aggregation_forms: ["explicit_month_list_metric"],
            supports_custom_aggregations: true,
            supports_raw_slices: false,
            available_years: [2022, 2022],
            default_years: [2022, 2022],
            default_months: [1, 3],
            postprocess_metrics: ["mean", "count_threshold"],
            postprocess_outputs: []
          }
        }
      ]
    });

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Start building my personalized dataset" }));
    fireEvent.click(screen.getByRole("button", { name: /Start new project/i }));

    await waitFor(() => {
      expect(screen.getByText("API ready")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "Sources" }));
    fireEvent.click(screen.getByText("Copernicus"));
    fireEvent.click(screen.getByRole("checkbox", { name: /copernicus_hrsi_snow/i }));
    fireEvent.click(screen.getByRole("button", { name: "Temporal" }));

    expect(screen.getByText("Input variable: snow_fraction")).toBeTruthy();
    expect(screen.queryByText("Base variables")).toBeNull();
  });
});
