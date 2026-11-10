"""Generate each scenario's network.net.xml from its plain XML node and edge files.

SUMO simulates on a compiled network file (.net.xml) that contains lane
geometry, junction internals, right-of-way rules and default traffic-light
programs. Writing that file by hand is error-prone, so the repository keeps
the short, readable node (.nod.xml) and edge (.edg.xml) descriptions and lets
SUMO's netconvert tool build the full network.

Usage:
    python scenarios/build_networks.py              # build all scenarios
    python scenarios/build_networks.py vehicle_control
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make the project importable

from envs.sumo_base import SCENARIO_DIR, find_sumo_tool  # noqa: E402

# scenario folder -> (node file, edge file)
SCENARIOS = {
    "simple_intersection": ("intersection.nod.xml", "intersection.edg.xml"),
    "vehicle_control": ("road.nod.xml", "road.edg.xml"),
}


def build_network(scenario: str) -> Path:
    folder = SCENARIO_DIR / scenario
    nodes, edges = SCENARIOS[scenario]
    output = folder / "network.net.xml"
    command = [
        find_sumo_tool("netconvert"),
        "--node-files", str(folder / nodes),
        "--edge-files", str(folder / edges),
        # No U-turn connections: they would add little and clutter the signal plan.
        "--no-turnarounds", "true",
        "--output-file", str(output),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="*", help=f"scenarios to build (default: all of {', '.join(SCENARIOS)})")
    args = parser.parse_args()
    unknown = set(args.scenarios) - set(SCENARIOS)
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(sorted(unknown))}")
    for scenario in args.scenarios or SCENARIOS:
        print(f"Built {build_network(scenario)}")


if __name__ == "__main__":
    main()
