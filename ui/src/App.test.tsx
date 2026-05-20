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
      ]
    }
  ],
  supported_metrics: ["mean", "sum"],
  supported_resampling: ["nearest", "average"],
  supported_stages: ["build"]
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

    render(<App />);

    expect(screen.getByText("Pirineus Raster Workbench")).toBeTruthy();

    await waitFor(() => {
      expect(screen.getByText("API ready")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "Sources" }));

    expect(screen.getByText("worldclim_cmip6_future")).toBeTruthy();
  });
});
