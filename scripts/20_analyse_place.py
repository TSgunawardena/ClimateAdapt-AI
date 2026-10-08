"""Command-line test for instant place analysis and the three-agent workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from climateadapt.agent import PlaceAnalysisOrchestrator  # noqa: E402
from climateadapt.place_analysis import PlaceAnalysisService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse a place from the precomputed Nelson V2 regional rasters."
    )
    parser.add_argument("place", nargs="?", default="Atawhai")
    parser.add_argument("--radius", type=int, choices=[500, 1000, 2000], default=1000)
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    arguments = parser.parse_args()

    if (arguments.latitude is None) != (arguments.longitude is None):
        parser.error("--latitude and --longitude must be supplied together.")

    service = PlaceAnalysisService(PROJECT_ROOT)
    place = service.resolve_place(
        arguments.place, arguments.latitude, arguments.longitude
    )
    result = service.analyse(place, arguments.radius)

    print("\n" + "=" * 78)
    print("CLIMATEADAPT AI – INSTANT PLACE ANALYSIS")
    print("=" * 78)
    print(f"Place:  {result['place']['display_name']}")
    print(f"Radius: {result['radius_metres']:,} metres")
    print(f"Status: {result['status']}")
    if result["status"] != "analysis_ready":
        print(f"Reason: {result.get('reason', 'Data unavailable')}")
        raise SystemExit(2)

    state = PlaceAnalysisOrchestrator(PROJECT_ROOT).run(result)
    print(f"Workflow status:       {state.status}")
    print(f"Evaluated land area:   {result['evaluated_land_area_square_kilometres']:.3f} km²")
    print(f"Susceptible area:      {result['susceptible_area_square_kilometres']:.3f} km²")
    print(
        "Susceptible proportion: "
        f"{result['susceptible_percentage_of_evaluated_land']:.2f}%"
    )
    print(f"Mean probability:      {result['mean_probability']:.4f}")
    print(f"Mean uncertainty:      {result['mean_uncertainty']:.4f}")
    print("Artifacts:")
    for name, path in state.artifacts.items():
        print(f"  {name}: {path}")
    print("=" * 78)
    print("Human review is required before further use.")


if __name__ == "__main__":
    main()
