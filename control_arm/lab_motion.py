"""Supervised incremental motion using strict servo feedback, never guessed poses."""
import sys
import threading
import math
from contextlib import contextmanager
from pathlib import Path
from lab_sessions import SessionError

# The safe zone lives with the platform, in learning_dashboards/scripts/robot (docs/safe-zone.md there).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'learning_dashboards' / 'scripts' / 'robot'))
try:
    import safe_zone
except ImportError:
    safe_zone = None  # Arming refuses without it.

UNITS_PER_DEGREE = 4095 / 360
JOINT_LIMITS = {
    'shoulder_pan': (-90, 90), 'shoulder_lift': (-30, 0),
    'elbow_flex': (0, 40), 'wrist_flex': (-30, 0),
    'wrist_roll': (-20, 20), 'gripper': (0, 60),
}


class ServoGate:
    """Also blocks legacy/background SDK writers during managed sessions."""
    READS = {'ReadPos', 'ReadSpeed', 'ReadMoving', 'ReadPosSpeed', 'read1ByteTxRx',
             'read2ByteTxRx', 'read4ByteTxRx', 'ping', 'getTxRxResult', 'getRxPacketError'}

    def __init__(self, sdk, sessions):
        self.sdk, self.sessions = sdk, sessions

    def __getattr__(self, name):
        value = getattr(self.sdk, name)
        if name in self.READS:
            return value
        if not callable(value):
            if self.sessions.lease or (self.sessions.motion and self.sessions.motion.managed):
                raise SessionError('Local servo access is locked by the live lab', 423)
            return value
        def guarded(*args, **kwargs):
            # Legacy callers may already hold the serial lock. Never wait here:
            # remote operations take the session lock before the serial lock.
            if not self.sessions.lock.acquire(blocking=False):
                raise SessionError('Robot control is busy', 423)
            try:
                motion = self.sessions.motion
                if (self.sessions.lease or (motion and motion.managed)) and not (motion and motion.permitted()):
                    raise SessionError('Local movement is locked by the live lab', 423)
                return value(*args, **kwargs)
            finally:
                self.sessions.lock.release()
        return guarded


class LabMotion:
    def __init__(self, robot, joint_ids, inversions, busy, clock, camera_ready=lambda: True, zone=safe_zone):
        self.robot, self.joint_ids, self.inversions = robot, joint_ids, inversions
        self.busy, self.clock = busy, clock
        self.camera_ready = camera_ready
        self.zone = zone
        self.managed = False
        self.armed = False
        self.fault = None
        self.bounds = {}
        self.positions = {}
        self.targets = {}
        self.last_move = float('-inf')
        self.local = threading.local()

    def permitted(self):
        return getattr(self.local, 'allowed', False)

    @contextmanager
    def permission(self):
        old = self.permitted()
        self.local.allowed = True
        try:
            yield
        finally:
            self.local.allowed = old

    @staticmethod
    def checked(result, name, position=False):
        if not isinstance(result, tuple) or len(result) != 3:
            raise SessionError('Invalid servo feedback: ' + name, 503)
        value, comm, error = result
        if comm != 0 or error != 0 or type(value) is not int or (position and not 0 <= value <= 4095):
            raise SessionError('Failed servo feedback: ' + name, 503)
        return value

    @staticmethod
    def degrees(raw):
        # The Play convention: raw 2048 is 0°, whatever the direction setting.
        return (raw - 2048) / UNITS_PER_DEGREE

    def zone_reason(self, measured, joint, target):
        """Why moving joint to target would leave the safe zone, or None."""
        now = {name: self.degrees(raw) for name, raw in measured.items()}
        goal = {joint: self.degrees(target)}
        # Other joints may still be travelling to earlier targets, so check from both ends of that travel.
        pending = dict(now, **{name: self.degrees(raw) for name, raw in self.targets.items() if name != joint})
        return (self.zone.check_move(now, goal) or self.zone.check_pose(pending)
                or self.zone.check_move(pending, goal))

    def connected(self):
        robot = self.robot()
        if not robot or not robot.connected or not robot.servo:
            raise SessionError('Connect the robot on the original Play page first', 409)
        return robot

    def arm(self):
        if self.armed:
            raise SessionError('Controls are already armed; end this session before changing the motion window')
        if self.busy():
            raise SessionError('Stop tracking, calibration, motion modes, teleoperation and simulation first')
        if not self.camera_ready():
            raise SessionError('Start the live camera and wait for a fresh image first')
        robot = self.connected()
        bounds, positions = {}, {}
        with robot._lock:
            ids = self.joint_ids()
            if len(ids) != 6 or len(set(ids.values())) != 6:
                raise SessionError('Six distinct joint IDs are required')
            for name, sid in ids.items():
                servo = robot.servo
                # Addresses come from the bundled sms_sts SDK memory table.
                mode = self.checked(servo.read1ByteTxRx(sid, 33), name)
                torque = self.checked(servo.read1ByteTxRx(sid, 40), name)
                moving = self.checked(servo.ReadMoving(sid), name)
                lo = self.checked(servo.read2ByteTxRx(sid, 9), name, True)
                hi = self.checked(servo.read2ByteTxRx(sid, 11), name, True)
                pos = self.checked(servo.ReadPos(sid), name, True)
                if mode != 0 or torque != 1 or moving != 0:
                    raise SessionError(f'{name}: position mode and powered, stationary servos are required (mode={mode}, torque={torque}, moving={moving})')
                if lo >= hi or not lo <= pos <= hi:
                    raise SessionError(f'{name}: hardware limits do not contain the current position (min={lo}, max={hi}, position={pos})')
                if self.inversions().get(name, 1) not in (-1, 1):
                    raise SessionError('Invalid direction setting: ' + name)
                if name not in JOINT_LIMITS:
                    raise SessionError('Joint has no approved limits: ' + name)
                # Match the existing Play /api/servo/move degree convention:
                # its two inversion operations cancel, leaving raw zero=2048.
                minimum, maximum = JOINT_LIMITS[name]
                bounds[name] = [max(lo, math.ceil(2048 + minimum * UNITS_PER_DEGREE)),
                                min(hi, math.floor(2048 + maximum * UNITS_PER_DEGREE))]
                if not bounds[name][0] <= pos <= bounds[name][1]:
                    raise SessionError(f'{name}: place the joint within {minimum}° to {maximum}° using local Play before arming')
                positions[name] = pos
        if self.zone is None:
            raise SessionError('The safe zone is not installed, so remote controls stay locked', 503)
        reason = self.zone.check_pose({name: self.degrees(raw) for name, raw in positions.items()})
        if reason:
            raise SessionError('Move the arm into the safe zone using local Play before arming. ' + reason)
        self.bounds, self.positions = bounds, positions
        self.targets = dict(positions)
        self.armed = self.managed = True
        self.fault = None
        self.last_move = float('-inf')

    def move(self, joint, delta=None, target_degrees=None):
        if not self.armed or self.fault:
            raise SessionError('Ask the administrator to arm supervised controls', 423)
        if not self.camera_ready():
            raise SessionError('Camera feedback is unavailable; wait for a fresh image', 409)
        if joint not in self.bounds or (delta is None) == (target_degrees is None):
            raise SessionError('Choose a joint and exactly one step or target', 400)
        if delta is not None and (type(delta) not in (int, float) or delta not in (-5, -1, 1, 5)):
            raise SessionError('Step must be -5, -1, +1 or +5 degrees', 400)
        if target_degrees is not None and (type(target_degrees) not in (int, float) or not math.isfinite(target_degrees)):
            raise SessionError('Target must be a finite angle', 400)
        if self.clock() - self.last_move < 0.5:
            raise SessionError('Wait before the next joint step', 429)
        robot = self.connected()
        with robot._lock:
            ids = self.joint_ids()
            sid = ids[joint]
            # The safe zone needs the whole arm, so read every joint, not just the one moving.
            measured = {name: self.checked(robot.servo.ReadPos(other), name, True) for name, other in ids.items()}
            pos = measured[joint]
            if target_degrees is None:
                target = pos + round(delta * UNITS_PER_DEGREE)
            else:
                minimum, maximum = JOINT_LIMITS[joint]
                if not minimum <= target_degrees <= maximum:
                    raise SessionError('Target is outside the approved joint limits', 400)
                # Quantize configured end points inward, then enforce EEPROM limits.
                target = max(math.ceil(2048 + minimum * UNITS_PER_DEGREE),
                             min(math.floor(2048 + maximum * UNITS_PER_DEGREE),
                                 round(2048 + target_degrees * UNITS_PER_DEGREE)))
            lo, hi = self.bounds[joint]
            if not lo <= pos <= hi or not lo <= target <= hi:
                raise SessionError('Step would leave the approved joint limits', 400)
            reason = self.zone_reason(measured, joint, target)
            if reason:
                raise SessionError(reason, 400)
            self.last_move = self.clock()
            with self.permission():
                result = robot.servo.WritePosEx(sid, target, 100, 10)
            if result != (0, 0):
                self.fault = 'Servo command failed; switch off the servo power and check the robot'
                raise SessionError(self.fault, 503)
            self.positions[joint] = pos
            self.targets[joint] = target

    def stop(self):
        was_armed = self.armed
        self.armed = False  # Latch closed even if feedback or holding fails.
        if not was_armed:
            return
        try:
            robot = self.connected()
            with robot._lock:
                # Read every joint successfully before issuing any hold target.
                positions = {name: self.checked(robot.servo.ReadPos(sid), name, True)
                             for name, sid in self.joint_ids().items()}
                with self.permission():
                    failures = []
                    for name, sid in self.joint_ids().items():
                        try:
                            ok = robot.servo.WritePosEx(sid, positions[name], 100, 10) == (0, 0)
                        except Exception:
                            ok = False
                        if not ok:
                            failures.append(name)
                    if failures:
                        raise SessionError('Hold command failed: ' + ', '.join(failures), 503)
                self.positions = positions
                self.targets = dict(positions)
        except Exception as error:
            self.fault = 'Software hold failed; switch off the servo power. ' + str(error)

    def return_local(self):
        self.stop()
        if self.fault:
            raise SessionError(self.fault, 503)
        self.managed = False
        self.bounds = {}

    def status(self):
        return {'armed': self.armed, 'managed': self.managed, 'fault': self.fault,
                'joints': list(self.bounds), 'stepDegrees': 1,
                'limitsDegrees': {name: list(limits) for name, limits in JOINT_LIMITS.items()},
                'positionsDegrees': {name: round((pos - 2048) / UNITS_PER_DEGREE, 1) for name, pos in self.positions.items()},
                'targetsDegrees': {name: round((pos - 2048) / UNITS_PER_DEGREE, 1) for name, pos in self.targets.items()}}
