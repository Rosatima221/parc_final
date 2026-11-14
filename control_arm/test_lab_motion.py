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

    def test_heartbeat_loss_holds_and_invalidates_session(self):
        self.arm()
        self.now += 10
        state = self.sessions.snapshot()
        self.assertIsNone(state['session'])
        self.assertEqual(len(self.sdk.writes), 6)
        with self.assertRaises(SessionError):
            self.move()

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
