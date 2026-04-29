import argparse

from src.io.config import load_yaml
from src.sources.worldclim.source import (
    prepare_worldclim_raw_data,
    prepare_worldclim_clipped_data,
    prepare_worldclim_features,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic raster feature pipeline."
    )

    parser.add_argument(
        "--project-config",
        default="configs/project.yaml",
        help="Path to project config YAML.",
    )

    parser.add_argument(
        "--source-config",
        default="configs/sources/worldclim/worldclim_v2_1_climate_normals.yaml",
        help="Path to source config YAML.",
    )

    parser.add_argument(
        "--stage",
        choices=["download", "clip", "build"],
        default="download",
        help="Pipeline stage to run.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_cfg = load_yaml(args.project_config)
    source_cfg = load_yaml(args.source_config)

    source = source_cfg["source"]

    provider = source["provider"]
    product = source["product"]

    print("==============================")
    print("Raster feature pipeline")
    print(f"Provider: {provider}")
    print(f"Product: {product}")
    print(f"Stage: {args.stage}")
    print("==============================")

    if provider == "worldclim":
        if args.stage == "download":
            zip_paths = prepare_worldclim_raw_data(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
            )

            print("==============================")
            print("Raw files ready")
            for path in zip_paths:
                print(f"  - {path}")
            print("==============================")

        elif args.stage == "clip":
            domains_cfg = source_cfg["domains"]
            clip_aoi_cfg = load_yaml(domains_cfg["clip_aoi_config"])

            clipped_paths = prepare_worldclim_clipped_data(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
                clip_aoi_cfg=clip_aoi_cfg,
            )

            print("==============================")
            print("Clipped files ready")
            print(f"Total files: {len(clipped_paths)}")
            for path in clipped_paths[:10]:
                print(f"  - {path}")
            if len(clipped_paths) > 10:
                print(f"  ... and {len(clipped_paths) - 10} more")
            print("==============================")

        elif args.stage == "build":
            domains_cfg = source_cfg["domains"]
            clip_aoi_cfg = load_yaml(domains_cfg["clip_aoi_config"])
            output_aoi_cfg = load_yaml(domains_cfg["output_aoi_config"])

            feature_paths = prepare_worldclim_features(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
                clip_aoi_cfg=clip_aoi_cfg,
                output_aoi_cfg=output_aoi_cfg,
            )

            print("==============================")
            print("Feature files ready")
            print(f"Total files: {len(feature_paths)}")
            for path in feature_paths[:10]:
                print(f"  - {path}")
            if len(feature_paths) > 10:
                print(f"  ... and {len(feature_paths) - 10} more")
            print("==============================")
            
        else:
            raise NotImplementedError(f"Stage not implemented yet: {args.stage}")

    else:
        raise NotImplementedError(
            f"No source connector implemented for provider={provider}, product={product}"
        )


if __name__ == "__main__":
    main()