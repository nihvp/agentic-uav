from agentic_uav.reasoning.scan import compact_nodes, merge_sightings, scan_headings


def node(i, label, prob, confirmed, pos=(1.0, 2.0, 0.5)):
    return {"id": i, "label": label, "existence_prob": prob, "position": list(pos),
            "attributes": ["state:confirmed" if confirmed else "state:tentative", "detector:color"]}


def test_scan_headings_full_turn_back_to_start():
    assert scan_headings(350.0) == [50.0, 110.0, 170.0, 230.0, 290.0, 350.0]
    assert len(scan_headings(0.0, 90.0)) == 4
    assert len(scan_headings(0.0, 1.0)) == 24        # step clamped to 15 deg


def test_compact_nodes_lists_confirmed_first():
    out = compact_nodes([node(1, "fire_hazard", 0.4, False), node(2, "survivor", 0.9, True)])
    assert [n["label"] for n in out] == ["survivor", "fire_hazard"]
    assert out[0]["confirmed"] and "attributes" not in out[0]


def test_merge_keeps_best_confidence_and_latest_position():
    seen = {}
    merge_sightings(seen, [node(2, "survivor", 0.95, True, (8.1, 3.9, 0.7))], 0.0)
    merge_sightings(seen, [node(2, "survivor", 0.5, False, (8.0, 4.0, 0.7))], 60.0)
    s = seen[2]
    assert s["existence_prob"] == 0.95 and s["confirmed"]
    assert s["position"] == [8.0, 4.0, 0.7] and s["seen_at_heading_deg"] == 60.0


def test_merge_of_compacted_nodes_keeps_confirmed():
    # The scan tool merges nodes that _settled_scene already compacted.
    seen = {}
    merge_sightings(seen, compact_nodes([node(6, "survivor", 0.99, True, (8.3, 4.3, 0.1))]), 299.1)
    assert seen[6]["confirmed"] and seen[6]["existence_prob"] == 0.99
