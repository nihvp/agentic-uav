from agentic_uav.reasoning.sightings import SightingLog


def n(label, pos, confirmed=True):
    return {"id": 0, "label": label, "position": list(pos), "existence_prob": 0.99, "confirmed": confirmed}


def test_repeat_sightings_merge_and_average():
    log = SightingLog()
    log.add([n("survivor", (8.0, 4.4, 0.3))])
    log.add([n("survivor", (7.8, 4.0, 0.3))])          # new scene-graph id, same object
    known = log.known()
    assert len(known) == 1 and known[0]["ref"] == "survivor-1" and known[0]["times_seen"] == 2
    assert known[0]["position"] == [7.9, 4.2, 0.3]


def test_landing_pad_seen_from_different_sides_is_one_object():
    log = SightingLog()
    for pos in [(-0.32, 6.78, 0), (-0.65, 9.51, 0), (0.95, 7.6, 0), (-1.77, 7.92, 0)]:
        log.add([n("landing_zone", pos)])
    assert [o["ref"] for o in log.known()] == ["landing_zone-1"]


def test_unconfirmed_ignored_and_distinct_objects_kept():
    log = SightingLog()
    log.add([n("fire_hazard", (11.2, 4.6, 0)), n("fire_hazard", (13.7, 6.8, 0)), n("survivor", (8, 4, 0), False)])
    assert [o["ref"] for o in log.known()] == ["fire_hazard-1", "fire_hazard-2"]


def test_near_reports_what_the_vehicle_is_above():
    log = SightingLog()
    log.add([n("survivor", (7.96, 4.43, 0.3)), n("landing_zone", (1.0, 6.8, 0))])
    assert log.near([7.78, 4.56, 5.0]) == [{"ref": "survivor-1", "label": "survivor", "horizontal_distance_m": 0.22}]
    assert log.near([4.0, 4.0, 5.0]) == []
    log.reset()
    assert log.known() == []
