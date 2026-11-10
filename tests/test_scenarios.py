"""Consistency checks on the scenario XML files (no SUMO needed)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
SCENARIOS = {
    "simple_intersection": ("intersection.nod.xml", "intersection.edg.xml"),
    "vehicle_control": ("road.nod.xml", "road.edg.xml"),
}


def load(path):
    return ET.parse(path).getroot()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_config_refers_to_existing_or_buildable_files(scenario):
    folder = SCENARIO_DIR / scenario
    config = load(folder / "simulation.sumocfg")
    assert config.find("./input/net-file").get("value") == "network.net.xml"  # written by build_networks.py
    for route_file in config.find("./input/route-files").get("value").split(","):
        assert (folder / route_file).is_file(), route_file
    assert config.find("./time/step-length").get("value") == "1"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_routes_use_defined_edges_and_types(scenario):
    folder = SCENARIO_DIR / scenario
    node_file, edge_file = SCENARIOS[scenario]
    nodes = {n.get("id") for n in load(folder / node_file).iter("node")}
    edges = {e.get("id"): e for e in load(folder / edge_file).iter("edge")}
    for edge in edges.values():
        assert edge.get("from") in nodes and edge.get("to") in nodes

    routes = load(folder / "traffic.rou.xml")
    vtypes = {t.get("id") for t in routes.iter("vType")}
    route_edges = {r.get("id"): r.get("edges").split() for r in routes.iter("route")}
    for edge_list in route_edges.values():
        assert set(edge_list) <= set(edges)
        for a, b in zip(edge_list, edge_list[1:]):
            assert edges[a].get("to") == edges[b].get("from"), f"{a} does not lead to {b}"

    departures = []
    for element in routes:
        if element.tag in ("flow", "vehicle"):
            assert element.get("route") in route_edges
            assert element.get("type") in vtypes
            departures.append(float(element.get("begin", element.get("depart"))))
    assert departures == sorted(departures), "SUMO skips vehicles that are not sorted by departure"


def test_intersection_is_signalised_and_every_movement_has_demand():
    nodes = load(SCENARIO_DIR / "simple_intersection" / "intersection.nod.xml")
    assert {n.get("id"): n.get("type") for n in nodes.iter("node")}["center"] == "traffic_light"
    routes = load(SCENARIO_DIR / "simple_intersection" / "traffic.rou.xml")
    assert len({f.get("route") for f in routes.iter("flow")}) == 12      # 4 approaches x 3 movements


def test_vehicle_scenario_defines_the_controlled_vehicle():
    routes = load(SCENARIO_DIR / "vehicle_control" / "traffic.rou.xml")
    ego = [v for v in routes.iter("vehicle") if v.get("id") == "ego"]
    assert len(ego) == 1
    controlled_type = next(t for t in routes.iter("vType") if t.get("id") == ego[0].get("type"))
    assert controlled_type.get("sigma") == "0"
