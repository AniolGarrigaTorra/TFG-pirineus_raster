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
    },
    {
      id: "copernicus_clms_hrvpp_vpp_laea",
      title: "CLMS HR-VPP vegetation phenology and productivity",
      provider: "copernicus_clms",
      product: "hrvpp_vpp_laea",
      config_path: "configs/sources/copernicus/copernicus_clms_hrvpp_vpp_laea.yaml",
      variables: [
        {
          name: "amplitude",
          kind: "variable",
          enabled_default: true,
          description: "Seasonal amplitude",
          temporal: {
            type: "yearly_static_collection",
            variable_pattern: "amplitude_{season}_{year}"
          }
        }
      ],
      dimensions: {
        growth_season: ["s1", "s2"]
      },
      dimension_context_keys: {
        growth_season: "season"
      },
      temporal: {
        kind: "yearly_static_collection",
        label: "Yearly static layers",
        default_output_mode: "supplied_layers",
        output_modes: ["supplied_layers", "aggregate"],
        aggregation_forms: ["year_range_metric"],
        supports_custom_aggregations: true,
        supports_raw_slices: false,
        default_years: [2017, 2024],
        available_years: [2017, 2024],
        temporal_layers: {
          years: [2020, 2021]
        }
      }
    }
  ],
  supported_metrics: ["mean", "sum"],
  supported_resampling: ["nearest", "average"],
  supported_stages: ["download", "clip", "build", "all"]
};

function mockApi(nextCatalog: WorkbenchCatalog) {
  class MockImage {
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    set src(_value: string) {
      window.setTimeout(() => this.onload?.(), 0);
    }
  }
  vi.stubGlobal("Image", MockImage);
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/backgrounds/manifest.json")) {
      return {
        ok: true,
        text: async () => JSON.stringify({ images: ["/backgrounds/test.jpg"] }),
        json: async () => ({ images: ["/backgrounds/test.jpg"] })
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
  it("opens the feature-oriented project setup", async () => {
    mockApi(catalog);

    render(<App />);

    expect(await screen.findByText("Welcome to Pirineus Raster")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Start building my personalized dataset" }));

    expect(screen.getByText("Start new project")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Start new project/i }));

    expect(screen.getByRole("heading", { name: "Project Setup" })).toBeTruthy();

    await waitFor(() => {
      expect(screen.getByText("API ready")).toBeTruthy();
    });

    expect((screen.getByLabelText("all") as HTMLInputElement).checked).toBe(true);
    expect((screen.getByLabelText("build") as HTMLInputElement).checked).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Start creating features" }));
    expect(screen.getByRole("heading", { name: "Feature Builder" })).toBeTruthy();
    expect(screen.getByRole("button", { name: /Build custom feature/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /Add official source layers/i })).toBeTruthy();
  });

  it("keeps AOI creation in the dedicated AOI workflow", async () => {
    mockApi(catalog);

    render(<App />);
    await screen.findByText("Welcome to Pirineus Raster");
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
    await screen.findByText("Welcome to Pirineus Raster");
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

  it("adds official source layers as final features and renders feature YAML", async () => {
    mockApi(catalog);

    render(<App />);
    await screen.findByText("Welcome to Pirineus Raster");
    fireEvent.click(screen.getByRole("button", { name: "Start building my personalized dataset" }));
    fireEvent.click(screen.getByRole("button", { name: /Start new project/i }));

    await waitFor(() => {
      expect(screen.getByText("API ready")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "Start creating features" }));
    fireEvent.click(screen.getByRole("button", { name: /Add official source layers/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: /tmin/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: "ACCESS-CM2" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "ssp126" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2021-2040" }));
    fireEvent.click(screen.getByRole("button", { name: "Add aggregation" }));
    fireEvent.click(screen.getByRole("button", { name: /Add 1 feature/i }));

    expect(screen.getByText(/1 cards/i)).toBeTruthy();
    expect(screen.getByText(/tmin · source_layer/i)).toBeTruthy();

    const reviewButtons = screen.getAllByRole("button", { name: "Review" });
    fireEvent.click(reviewButtons[reviewButtons.length - 1]);

    const yaml = screen.getByText((content) => content.includes("features:"));
    expect(yaml.textContent).toContain("build_type: source_layer");
    expect(yaml.textContent).not.toContain("\nsources:");
  });

  it("requires yearly dimensions and temporal layers before adding HR-VPP outputs", async () => {
    mockApi(catalog);

    render(<App />);
    await screen.findByText("Welcome to Pirineus Raster");
    fireEvent.click(screen.getByRole("button", { name: "Start building my personalized dataset" }));
    fireEvent.click(screen.getByRole("button", { name: /Start new project/i }));

    await waitFor(() => {
      expect(screen.getByText("API ready")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "Start creating features" }));
    fireEvent.click(screen.getByRole("button", { name: /Add official source layers/i }));
    fireEvent.click(screen.getByRole("button", { name: /copernicus_clms_hrvpp_vpp_laea/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Seasonal amplitude/i }));

    expect(screen.getByRole("button", { name: /Add 1 feature/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("checkbox", { name: "s1" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "2020" }));
    fireEvent.click(screen.getByRole("button", { name: /Add 1 feature/i }));

    const reviewButtons = screen.getAllByRole("button", { name: "Review" });
    fireEvent.click(reviewButtons[reviewButtons.length - 1]);

    const yaml = screen.getByText((content) => content.includes("features:"));
    expect(yaml.textContent).toContain("amplitude_s1_2020");
    expect(yaml.textContent).toContain("output_mode: supplied_layers");
    expect(yaml.textContent).not.toContain("output_mode: static");
  });
});
