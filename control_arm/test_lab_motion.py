"""Motion safety tests with a fake SDK only; never opens serial hardware."""
import threading
import unittest
from types import SimpleNamespace
from lab_motion import LabMotion, ServoGate, JOINT_LIMITS, UNITS_PER_DEGREE
import lab_motion
import math
from lab_sessions import LabSessions, SessionError

ADMIN = {'id': 'admin', 'role': 'admin'}
OWNER = {'id': 'owner', 'role': 'learner', 'name': 'Learner'}
OTHER = {'id': 'other', 'role': 'learner'}


class FakeSDK:
    def __init__(self):
        self.positions = {sid: 2048 for sid in range(1, 7)}
        self.writes = []
        self.failed_read = None
        self.mode = 0
        self.write_error = False

    def ReadPos(self, sid):
        return self.positions[sid], (-1 if self.failed_read == sid else 0), 0

    def ReadMoving(self, sid):
        return 0, 0, 0

    def read1ByteTxRx(self, sid, address):
        return (self.mode if address == 33 else 1), 0, 0

    def read2ByteTxRx(self, sid, address):
        return (100 if address == 9 else 4000), 0, 0

    def WritePosEx(self, sid, target, speed, acc):
        self.writes.append((sid, target, speed, acc))
        if self.write_error:
            return -1, 0
        self.positions[sid] = target
        return 0, 0


class FakeZone:
    def __init__(self, pose=None, move=None):
        self.pose_reason, self.move_reason = pose, move
        self.moves = []

    def check_pose(self, pose):
        return self.pose_reason

    def check_move(self, start, target):
        self.moves.append((dict(start), dict(target)))
        return self.move_reason


def raw(degrees):
    return 2048 + round(degrees * UNITS_PER_DEGREE)


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.now = 100
        self.sessions = LabSessions(clock=lambda: self.now)
        self.ids = {name: index + 1 for index, name in enumerate(
            ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'])}
        self.sdk = FakeSDK()
        self.robot = SimpleNamespace(connected=True, _lock=threading.Lock(), servo=ServoGate(self.sdk, self.sessions))
        self.camera = True
        self.motion = LabMotion(lambda: self.robot, lambda: self.ids,
                                lambda: {'shoulder_pan': -1}, lambda: False, lambda: self.now,
                                lambda: self.camera)
        self.sessions.motion = self.motion
        self.sid = self.sessions.assign(ADMIN, OWNER, 1)['session']['id']

    def arm(self):
        self.sessions.update(ADMIN, self.sid, 'arm', {'workspaceClear': True})
        self.sessions.update(OWNER, self.sid, 'heartbeat')

    def move(self, joint='shoulder_pan', delta=1):
        return self.sessions.update(OWNER, self.sid, 'move', {'joint': joint, 'delta': delta})

    def test_arm_reads_configuration_without_moving(self):
        self.arm()
        self.assertEqual(self.sdk.writes, [])
        self.assertTrue(self.sessions.snapshot()['controlEnabled'])

    def test_only_admin_can_arm_and_confirmation_required(self):
        for actor, data in [(OWNER, {'workspaceClear': True}), (ADMIN, {})]:
            with self.assertRaises(SessionError):
                self.sessions.update(actor, self.sid, 'arm', data)
        self.assertEqual(self.sdk.writes, [])

    def test_only_joined_owner_can_move(self):
        self.sessions.update(ADMIN, self.sid, 'arm', {'workspaceClear': True})
        for actor in (OWNER, OTHER, ADMIN):
            with self.assertRaises(SessionError):
                self.sessions.update(actor, self.sid, 'move', {'joint': 'gripper', 'delta': 1})
        self.assertEqual(self.sdk.writes, [])

    def test_finite_step_and_direction_with_rate_limit(self):
        self.arm()
        self.move()
        self.assertEqual(self.sdk.writes[-1], (1, 2059, 100, 10))
        with self.assertRaises(SessionError):
            self.move()
        self.assertEqual(len(self.sdk.writes), 1)

    def test_can_move_beyond_old_five_degree_window(self):
        self.arm()
        for _ in range(10):
            self.now += 0.6
            self.move()
        with self.assertRaises(SessionError):
            self.motion.arm()
        self.assertEqual(len(self.sdk.writes), 10)

    def test_exact_approved_limits_for_every_joint(self):
        self.arm()
        self.assertEqual(JOINT_LIMITS, {
            'shoulder_pan': (-90, 90), 'shoulder_lift': (-30, 0),
            'elbow_flex': (0, 40), 'wrist_flex': (-30, 0),
            'wrist_roll': (-20, 20), 'gripper': (0, 60)})
        for name, (low, high) in JOINT_LIMITS.items():
            lo, hi = self.motion.bounds[name]
            self.assertEqual(lo, math.ceil(2048 + low * UNITS_PER_DEGREE))
            self.assertEqual(hi, math.floor(2048 + high * UNITS_PER_DEGREE))
            for position, delta in [(lo, -1), (hi, 1)]:
                self.sdk.positions[self.ids[name]] = position
                self.now += 1
                self.sessions.update(OWNER, self.sid, 'heartbeat')
                with self.assertRaises(SessionError):
                    self.move(name, delta)
        self.assertEqual(self.sdk.writes, [])

    def test_starting_outside_approved_range_does_not_move(self):
        self.sdk.positions[self.ids['shoulder_lift']] = 2060
        with self.assertRaises(SessionError):
            self.arm()
        self.assertFalse(self.motion.armed)
        self.assertEqual(self.sdk.writes, [])

    def test_hardware_limits_still_override_approved_range(self):
        original = self.sdk.read2ByteTxRx
        self.sdk.read2ByteTxRx = lambda sid, addr: (2050, 0, 0) if sid == 1 and addr == 11 else original(sid, addr)
        self.arm()
        self.assertEqual(self.motion.bounds['shoulder_pan'][1], 2050)
        with self.assertRaises(SessionError):
            self.move()
        self.assertEqual(self.sdk.writes, [])

    def test_invalid_steps_and_joint_never_write(self):
        self.arm()
        for joint, delta in [('missing', 1), ('gripper', 90), ('gripper', True), ('gripper', float('nan'))]:
            with self.assertRaises(SessionError):
                self.move(joint, delta)
        self.assertEqual(self.sdk.writes, [])

    def test_bad_feedback_never_becomes_guessed_position(self):
        self.arm()
        self.sdk.failed_read = 1
        with self.assertRaises(SessionError):
            self.move()
        self.assertEqual(self.sdk.writes, [])
        self.assertFalse(self.motion.armed)
        self.assertIsNotNone(self.motion.fault)

    def test_camera_loss_blocks_commands(self):
        self.arm()
        self.camera = False
        with self.assertRaises(SessionError):
            self.move()
        self.assertEqual(self.sdk.writes, [])

    def test_five_degree_steps(self):
        self.arm()
        self.move('shoulder_pan', 5)
        self.assertEqual(self.sdk.writes[-1], (1, 2105, 100, 10))

    def test_slider_target_and_feedback_are_distinct(self):
        self.arm()
        self.sessions.update(OWNER, self.sid, 'move', {'joint': 'elbow_flex', 'target': 20})
        self.assertEqual(self.sdk.writes[-1], (3, 2276, 100, 10))
        status = self.motion.status()
        self.assertEqual(status['positionsDegrees']['elbow_flex'], 0)
        self.assertEqual(status['targetsDegrees']['elbow_flex'], 20)

    def test_slider_all_range_endpoints_round_inward(self):
        self.arm()
        # Some combinations of end points come near the table; the safe zone has its own tests.
        self.motion.zone = FakeZone()
        for name in self.ids:
            for index, angle in enumerate(JOINT_LIMITS[name]):
                self.now += 1
                self.sessions.update(OWNER, self.sid, 'heartbeat')
                self.sessions.update(OWNER, self.sid, 'move', {'joint': name, 'target': angle})
                target = self.sdk.writes[-1][1]
                self.assertEqual(target, self.motion.bounds[name][index])

    def test_invalid_slider_targets_and_ambiguous_requests_never_write(self):
        self.arm()
        for target in (-91, 91, True, float('nan'), float('inf'), '30'):
            with self.assertRaises(SessionError):
                self.sessions.update(OWNER, self.sid, 'move', {'joint': 'shoulder_pan', 'target': target})
        with self.assertRaises(SessionError):
            self.sessions.update(OWNER, self.sid, 'move', {'joint': 'shoulder_pan', 'target': 10, 'delta': 1})
        self.assertEqual(self.sdk.writes, [])

    def test_slider_does_not_bypass_owner_or_camera(self):
        self.arm()
        with self.assertRaises(SessionError):
            self.sessions.update(OTHER, self.sid, 'move', {'joint': 'gripper', 'target': 30})
        self.camera = False
        with self.assertRaises(SessionError):
            self.sessions.update(OWNER, self.sid, 'move', {'joint': 'gripper', 'target': 30})
        self.assertEqual(self.sdk.writes, [])

    def test_wrong_servo_mode_prevents_arming(self):
        self.sdk.mode = 1
        with self.assertRaises(SessionError):
            self.arm()
        self.assertFalse(self.motion.armed)
        self.assertEqual(self.sdk.writes, [])

    def test_stop_holds_measured_positions_and_locks_local(self):
        self.arm()
        self.sessions.update(OWNER, self.sid, 'stop')
        self.assertEqual(len(self.sdk.writes), 6)
        self.assertTrue(all(target == 2048 for _, target, _, _ in self.sdk.writes))
        self.assertFalse(self.motion.armed)
        self.assertIsNone(self.sessions.lease)
        with self.assertRaises(SessionError):
            self.robot.servo.WritePosEx(1, 3000, 500, 50)
        self.sessions.update(ADMIN, None, 'local')
        self.robot.servo.WritePosEx(1, 2048, 100, 10)

    def test_heartbeat_loss_holds_the_arm_and_keeps_it_armed_until_the_owner_returns(self):
        self.arm()
        self.now += 10
        state = self.sessions.snapshot()
        self.assertTrue(state['session']['disconnected'])
        self.assertEqual(len(self.sdk.writes), 6)
        self.assertTrue(self.motion.armed)
        self.assertFalse(state['controlEnabled'])
        with self.assertRaises(SessionError):
            self.move()
        self.assertEqual(len(self.sdk.writes), 6)
        self.sessions.update(OWNER, self.sid, 'heartbeat')
        self.assertTrue(self.sessions.snapshot()['controlEnabled'])
        self.now += 1
        self.move()
        self.assertEqual(len(self.sdk.writes), 7)

    def test_failed_hold_on_disconnect_disarms(self):
        self.arm()
        self.sdk.write_error = True
        self.now += 10
        self.sessions.snapshot()
        self.assertFalse(self.motion.armed)
        self.assertIsNotNone(self.motion.fault)

    def test_failed_hold_latches_closed(self):
        self.arm()
        self.sdk.write_error = True
        self.sessions.update(ADMIN, self.sid, 'stop')
        self.assertFalse(self.motion.armed)
        self.assertIsNotNone(self.motion.fault)
        self.assertEqual(len(self.sdk.writes), 6)
        with self.assertRaises(SessionError):
            self.sessions.update(ADMIN, None, 'local')

    def test_legacy_sdk_commands_blocked_during_assignment(self):
        with self.assertRaises(SessionError):
            self.robot.servo.WritePosEx(1, 3000, 500, 50)
        self.assertEqual(self.sdk.writes, [])

    FOLDED = {'shoulder_pan': -92, 'shoulder_lift': -111, 'elbow_flex': 98, 'wrist_flex': 76,
              'wrist_roll': -7, 'gripper': -1}

    def fold(self, **changes):
        for name, degrees in dict(self.FOLDED, **changes).items():
            self.sdk.positions[self.ids[name]] = raw(degrees)

    def start(self):
        self.sessions.update(ADMIN, self.sid, 'start', {'workspaceClear': True})
        deadline = lab_motion.time.monotonic() + 5
        while self.motion.start_state['running'] and lab_motion.time.monotonic() < deadline:
            lab_motion.time.sleep(0.01)

    def test_start_pose_moves_stage_by_stage_to_the_start_pose(self):
        self.fold()
        self.start()
        self.assertTrue(self.motion.start_state['done'], self.motion.fault)
        self.assertFalse(self.motion.start_state['running'])
        for name, degrees in lab_motion.START_POSE.items():
            self.assertEqual(self.sdk.positions[self.ids[name]], raw(degrees), name)
        order = [sid for sid, *_ in self.sdk.writes]
        names = {sid: name for name, sid in self.ids.items()}
        seen = []
        for sid in order:
            if names[sid] not in seen:
                seen.append(names[sid])
        # The wrist lifts away from the shoulder first, then the elbow unfolds, then the shoulder lifts.
        self.assertEqual(seen[:3], ['wrist_flex', 'elbow_flex', 'shoulder_lift'])
        self.assertTrue(all(speed == lab_motion.START_SPEED for _, _, speed, _ in self.sdk.writes))
        # Each step is a few degrees at most.
        previous = {}
        for sid, target, _, _ in self.sdk.writes:
            if sid in previous:
                self.assertLessEqual(abs(target - previous[sid]), raw(lab_motion.START_STEP_DEGREES + 1) - 2048)
            previous[sid] = target
        self.assertTrue(self.motion.managed)
        self.assertFalse(self.motion.armed)
        self.assertIsNone(self.motion.fault)

    def test_start_pose_continues_from_part_of_the_way(self):
        # A move that stopped after the wrist and part of the elbow carries on from there.
        self.fold(wrist_flex=-15, elbow_flex=60)
        self.start()
        self.assertTrue(self.motion.start_state['done'], self.motion.fault)
        for name, degrees in lab_motion.START_POSE.items():
            self.assertEqual(self.sdk.positions[self.ids[name]], raw(degrees), name)

    def test_start_pose_refuses_a_less_folded_elbow(self):
        self.fold(elbow_flex=80)
        with self.assertRaises(SessionError):
            self.start()
        self.assertEqual(self.sdk.writes, [])

    def test_start_pose_then_arm_controls(self):
        self.fold()
        self.start()
        self.arm()
        self.assertTrue(self.sessions.snapshot()['controlEnabled'])

    def test_start_pose_needs_admin_and_confirmation(self):
        self.fold()
        for actor, data in [(OWNER, {'workspaceClear': True}), (ADMIN, {})]:
            with self.assertRaises(SessionError):
                self.sessions.update(actor, self.sid, 'start', data)
        self.assertEqual(self.sdk.writes, [])

    def test_start_pose_refuses_when_the_arm_is_not_at_the_rest_pose(self):
        self.fold(elbow_flex=50)
        with self.assertRaises(SessionError) as refused:
            self.start()
        self.assertIn('folded rest pose', refused.exception.message)
        self.assertEqual(self.sdk.writes, [])
        self.assertFalse(self.motion.managed)

    def test_start_pose_says_when_the_arm_is_already_inside_the_ranges(self):
        with self.assertRaises(SessionError) as refused:
            self.start()
        self.assertIn('already inside', refused.exception.message)

    def test_start_pose_refuses_with_a_bad_servo_reading(self):
        self.fold()
        self.sdk.failed_read = self.ids['wrist_flex']
        with self.assertRaises(SessionError):
            self.start()
        self.assertEqual(self.sdk.writes, [])

    def test_stop_during_the_start_move_holds_the_arm(self):
        self.fold()
        proceed = threading.Event()
        original = self.sdk.WritePosEx
        def slow(sid, target, speed, acc):
            proceed.wait(2)
            return original(sid, target, speed, acc)
        self.sdk.WritePosEx = slow
        self.sessions.update(ADMIN, self.sid, 'start', {'workspaceClear': True})
        self.assertTrue(self.motion.start_state['running'])
        stopper = threading.Thread(target=lambda: self.sessions.update(ADMIN, self.sid, 'stop'))
        stopper.start()
        lab_motion.time.sleep(0.05)
        proceed.set()
        stopper.join(timeout=3)
        deadline = lab_motion.time.monotonic() + 2
        while self.motion.starting and lab_motion.time.monotonic() < deadline:
            lab_motion.time.sleep(0.01)
        self.assertFalse(self.motion.start_state['running'])
        self.assertFalse(self.motion.start_state['done'])
        # The arm is nowhere near the start pose, and every joint was told to stay put.
        self.assertNotEqual(self.sdk.positions[self.ids['elbow_flex']], raw(lab_motion.START_POSE['elbow_flex']))
        self.assertGreaterEqual(len(self.sdk.writes), 6)

    def test_start_move_that_the_arm_does_not_follow_is_stopped(self):
        self.fold()
        self.sdk.WritePosEx = lambda sid, target, speed, acc: (0, 0)  # accepted, but the arm never moves
        original = lab_motion.ARRIVE_SECONDS
        lab_motion.ARRIVE_SECONDS = 0.2
        try:
            self.start()
        finally:
            lab_motion.ARRIVE_SECONDS = original
        self.assertFalse(self.motion.start_state['running'])
        self.assertFalse(self.motion.start_state['done'])
        self.assertIn('stopped following', self.motion.fault)
        self.assertIn('wrist flex', self.motion.fault)

    def stall_start(self, volts):
        self.fold()
        self.sdk.WritePosEx = lambda sid, target, speed, acc: (0, 0)  # accepted, but the arm never moves
        self.sdk.read1ByteTxRx = lambda sid, address: ({33: 0, lab_motion.VOLTAGE: volts}.get(address, 1), 0, 0)
        original = lab_motion.ARRIVE_SECONDS
        lab_motion.ARRIVE_SECONDS = 0.2
        try:
            self.start()
        finally:
            lab_motion.ARRIVE_SECONDS = original

    def test_start_stop_names_a_low_servo_supply(self):
        self.stall_start(53)
        self.assertIn('stopped following', self.motion.fault)
        self.assertIn('5.3 V', self.motion.fault)

    def test_start_stop_says_nothing_about_a_good_supply(self):
        self.stall_start(74)
        self.assertIn('stopped following', self.motion.fault)
        self.assertNotIn('supply', self.motion.fault)

    def test_stopped_start_pose_can_be_handed_back_to_local_control(self):
        self.stall_start(74)
        self.sessions.update(ADMIN, self.sid, 'release')
        self.sessions.update(ADMIN, None, 'local')
        self.assertIsNone(self.motion.fault)
        self.assertFalse(self.motion.managed)

    def test_failed_hold_still_blocks_local_control(self):
        self.stall_start(74)
        self.motion.fault = 'Software hold failed; switch off the servo power. bus error'
        self.sessions.update(ADMIN, self.sid, 'release')
        with self.assertRaises(SessionError):
            self.sessions.update(ADMIN, None, 'local')
        self.assertTrue(self.motion.managed)

    def test_status_reports_servo_health(self):
        self.sdk.read1ByteTxRx = lambda sid, address: ({lab_motion.VOLTAGE: 74, lab_motion.TEMPERATURE: 31,
                                                        lab_motion.STATUS: 0b100001}.get(address, 0), 0, 0)
        self.sdk.read2ByteTxRx = lambda sid, address: ((1 << 10) | 250 if address == lab_motion.LOAD else 0, 0, 0)
        self.motion.health_at = float('-inf')  # skip the reading cached when the session started
        elbow = self.motion.status()['health']['elbow_flex']
        self.assertEqual(elbow, {'volts': 7.4, 'celsius': 31, 'load': 25.0, 'errors': ['voltage', 'overload']})
        self.assertEqual(self.sdk.writes, [])

    def test_status_skips_health_when_the_bus_is_busy(self):
        self.motion.health, self.motion.health_at = {}, float('-inf')
        with self.robot._lock:
            self.assertEqual(self.motion.status()['health'], {})

    def test_connect_goes_through_the_controller_and_reports_failure(self):
        results = [(True, 'Connected'), (False, 'No such port')]
        self.motion.connect_robot = lambda: results.pop(0)
        self.sessions.update(ADMIN, self.sid, 'connect')
        with self.assertRaises(SessionError) as refused:
            self.sessions.update(ADMIN, self.sid, 'connect')
        self.assertIn('No such port', refused.exception.message)
        with self.assertRaises(SessionError):
            self.sessions.update(OWNER, self.sid, 'connect')

    def test_real_safe_zone_is_loaded(self):
        self.assertIsNotNone(lab_motion.safe_zone)
        self.assertIs(self.motion.zone, lab_motion.safe_zone)

    def test_safe_zone_refusal_never_writes_or_faults(self):
        self.arm()
        self.motion.zone = FakeZone(move='That would bring the gripper within 3 cm of the table.')
        with self.assertRaises(SessionError) as caught:
            self.move()
        self.assertEqual(caught.exception.status, 400)
        self.assertIn('table', caught.exception.message)
        self.assertEqual(self.sdk.writes, [])
        self.assertTrue(self.motion.armed)
        self.assertIsNone(self.motion.fault)

    def test_safe_zone_sees_the_whole_measured_arm(self):
        self.arm()
        zone = FakeZone()
        self.motion.zone = zone
        self.sdk.positions[self.ids['elbow_flex']] = raw(20)
        self.move('shoulder_pan', 5)
        start, goal = zone.moves[0]
        self.assertEqual(set(start), set(self.ids))
        self.assertAlmostEqual(start['elbow_flex'], 20, delta=0.1)
        self.assertEqual(list(goal), ['shoulder_pan'])
        self.assertAlmostEqual(goal['shoulder_pan'], 5, delta=0.1)

    def test_joint_still_travelling_is_checked_at_its_target(self):
        self.arm()
        zone = FakeZone()
        self.motion.zone = zone
        self.sessions.update(OWNER, self.sid, 'move', {'joint': 'elbow_flex', 'target': 20})
        self.sdk.positions[self.ids['elbow_flex']] = 2048  # still on its way
        self.now += 1
        self.move('shoulder_pan', 1)
        self.assertEqual(sorted(round(start['elbow_flex']) for start, _ in zone.moves[-2:]), [0, 20])

    def test_arming_outside_safe_zone_is_refused(self):
        self.motion.zone = FakeZone(pose='That would bring the wrist too close to the arm\'s base.')
        with self.assertRaises(SessionError) as caught:
            self.arm()
        self.assertIn('safe zone', caught.exception.message)
        self.assertFalse(self.motion.armed)
        self.assertEqual(self.sdk.writes, [])

    def test_missing_safe_zone_keeps_controls_locked(self):
        self.motion.zone = None
        with self.assertRaises(SessionError) as caught:
            self.arm()
        self.assertEqual(caught.exception.status, 503)
        self.assertFalse(self.motion.armed)

    def test_real_safe_zone_refuses_gripper_near_table(self):
        # Allowed with the wrist at -30°; straightening it to 0° brings the gripper within 3 cm of the table.
        for name, degrees in {'shoulder_lift': 0, 'elbow_flex': 40, 'wrist_flex': -29.9, 'wrist_roll': -19.9}.items():
            self.sdk.positions[self.ids[name]] = raw(degrees)
        self.arm()
        with self.assertRaises(SessionError) as caught:
            self.sessions.update(OWNER, self.sid, 'move', {'joint': 'wrist_flex', 'target': 0})
        self.assertIn('table', caught.exception.message)
        self.assertEqual(self.sdk.writes, [])
        self.sessions.update(OWNER, self.sid, 'move', {'joint': 'wrist_flex', 'target': -25})
        self.assertEqual(len(self.sdk.writes), 1)

    def test_legacy_lock_contention_does_not_deadlock(self):
        done = threading.Event()
        def writer():
            with self.robot._lock:
                try:
                    self.robot.servo.WritePosEx(1, 3000, 500, 50)
                except SessionError:
                    done.set()
        with self.sessions.lock:
            thread = threading.Thread(target=writer)
            thread.start()
            self.assertTrue(done.wait(1))
        thread.join(timeout=1)


if __name__ == '__main__':
    unittest.main()
