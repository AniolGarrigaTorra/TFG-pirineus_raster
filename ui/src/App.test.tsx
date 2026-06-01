import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the workbench with catalog sources", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      text: async () => JSON.stringify(catalog)
    } as Response));

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
});
