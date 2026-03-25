from __future__ import annotations

"""
Example downloader for Copernicus Marine subsets.

This requires:
1. A Copernicus Marine account.
2. `copernicusmarine` installed in the active environment.
3. Either interactive login or environment variables supported by the toolbox.

Example:
    pip install copernicusmarine
    python src/copernicus_subset_example.py
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"


def main() -> None:
    try:
        import copernicusmarine
    except Exception as exc:
        raise SystemExit(
            "copernicusmarine is not installed. Install it with `pip install copernicusmarine`."
        ) from exc

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # Physics subset for Indian waters.
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["so", "thetao", "mlotst"],
        minimum_longitude=65,
        maximum_longitude=95,
        minimum_latitude=5,
        maximum_latitude=25,
        start_datetime="2013-01-01T00:00:00",
        end_datetime="2024-12-31T23:59:59",
        output_filename="copernicus_physics.nc",
        output_directory=str(EXTERNAL_DIR),
        force_download=True,
    )

    # Chlorophyll subset for Indian waters.
    copernicusmarine.subset(
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D",
        variables=["CHL"],
        minimum_longitude=65,
        maximum_longitude=95,
        minimum_latitude=5,
        maximum_latitude=25,
        start_datetime="2013-01-01T00:00:00",
        end_datetime="2024-12-31T23:59:59",
        output_filename="copernicus_chlorophyll.nc",
        output_directory=str(EXTERNAL_DIR),
        force_download=True,
    )

    print(f"Saved Copernicus subsets into {EXTERNAL_DIR}")


if __name__ == "__main__":
    main()
