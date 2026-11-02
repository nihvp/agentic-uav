from agentic_uav.action.offboard import (
    ARMING_STATE_ARMED, CMD_COMPONENT_ARM_DISARM, CMD_DO_SET_MODE, CMD_NAV_RETURN_TO_LAUNCH,
    NAV_STATE_OFFBOARD, ControllerConfig, OffboardController, State,
)

NAV_STATE_POSCTL = 2


class FakePX4:
    """Moves the vehicle straight to the setpoint and honours mode/arm commands."""

    def __init__(self, ctrl: OffboardController):
        self.ctrl = ctrl
        self.pos = (0.0, 0.0, 0.0)
        self.armed = False
        self.offboard = False
        self.sync()

    def sync(self):
        self.ctrl.update_position(self.pos, 0.0, True)
        self.ctrl.update_status(ARMING_STATE_ARMED if self.armed else 1,
                                NAV_STATE_OFFBOARD if self.offboard else NAV_STATE_POSCTL)

    def step(self, n=1):
        for _ in range(n):
            out = self.ctrl.tick()
            cmds = {c.command for c in out.commands}
            if CMD_DO_SET_MODE in cmds:
                self.offboard = True
            if CMD_COMPONENT_ARM_DISARM in cmds:
                self.armed = True
            if CMD_NAV_RETURN_TO_LAUNCH in cmds:
                self.offboard = False
            if out.setpoint_ned is not None and self.offboard:
                self.pos = out.setpoint_ned
            self.sync()


def make():
    ctrl = OffboardController(ControllerConfig(preroll_ticks=3))
    return ctrl, FakePX4(ctrl)


def test_takeoff_then_navigate():
    ctrl, px4 = make()
    assert ctrl.takeoff(5.0)[0]
    px4.step(10)
    assert ctrl.state == State.HOLDING
    assert px4.pos == (0.0, 0.0, -5.0)

    ok, _, seq = ctrl.go_to((8.0, 4.0, 5.0))
    assert ok and ctrl.goal_result(seq) is None
    px4.step(2)
    assert ctrl.goal_result(seq) == (True, "Reached target.")
    assert px4.pos == (4.0, 8.0, -5.0)  # NED


def test_navigate_rejected_on_ground_and_outside_geofence():
    ctrl, px4 = make()
    assert not ctrl.go_to((1.0, 1.0, 5.0))[0]
    ctrl.takeoff(5.0)
    px4.step(10)
    assert not ctrl.go_to((500.0, 0.0, 5.0))[0]
    assert not ctrl.go_to((1.0, 0.0, 0.2))[0]


def test_external_takeover_aborts_goal():
    ctrl, px4 = make()
    ctrl.takeoff(5.0)
    px4.step(10)
    ctrl.cfg.acceptance_radius = 0.01
    _, _, seq = ctrl.go_to((8.0, 4.0, 5.0))
    px4.offboard = False  # pilot / failsafe switches mode before the vehicle moves
    px4.sync()
    px4.step()
    assert ctrl.state == State.EXTERNAL
    assert ctrl.goal_result(seq)[0] is False


def test_rtl_hands_off_to_px4():
    ctrl, px4 = make()
    ctrl.takeoff(5.0)
    px4.step(10)
    assert ctrl.rtl()[0]
    px4.step()
    assert ctrl.state == State.HANDED_OFF and not px4.offboard


def test_start_times_out_if_px4_refuses():
    ctrl = OffboardController(ControllerConfig(preroll_ticks=2, start_timeout_s=0.1, rate_hz=20.0))
    ctrl.update_position((0.0, 0.0, 0.0), 0.0, True)
    ctrl.takeoff()
    for _ in range(10):
        ctrl.tick()
    assert ctrl.state == State.IDLE and "did not accept" in ctrl.last_error
