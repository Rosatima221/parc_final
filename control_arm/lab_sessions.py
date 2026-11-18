"""Exclusive, expiring sessions with optional supervised physical motion."""
import hmac
import threading
import time
import uuid
from flask import Blueprint, jsonify, request


class SessionError(Exception):
    def __init__(self, message, status=409):
        self.message, self.status = message, status


class LabSessions:
    HEARTBEAT_SECONDS = 10

    def __init__(self, clock=time.monotonic, wall_clock=time.time):
        self.clock, self.wall_clock = clock, wall_clock
        self.lock = threading.RLock()
        self.lease = None
        self.deadline = 0
        self.heartbeat_at = None
        self.disconnected = False
        self.last_end_reason = None
        self._motion = None

    @property
    def motion(self):
        return self._motion

    @motion.setter
    def motion(self, motion):
        self._motion = motion
        if motion is not None:
            # Background motion takes this lock for each step, so STOP and the watchdog are never kept waiting.
            motion.lock = self.lock

    def _expire(self):
        if self.lease and self.clock() >= self.deadline:
            self._end('expired')
        elif self.lease and self.heartbeat_at is not None and self.clock() - self.heartbeat_at >= self.HEARTBEAT_SECONDS:
            # The page went quiet (a closed or sleeping tab, a lost connection). The slot goes on: hold the arm
            # where it is and let the owner come back. Nothing moves until they rejoin.
            self.heartbeat_at = None
            self.disconnected = True
            if self.motion:
                self.motion.hold()

    def _end(self, reason):
        if self.motion:
            self.motion.stop()
        self.lease = None
        self.heartbeat_at = None
        self.disconnected = False
        self.last_end_reason = reason

    def snapshot(self):
        with self.lock:
            self._expire()
            return {
                'session': dict(self.lease, joined=self.heartbeat_at is not None, disconnected=self.disconnected) if self.lease else None,
                'controlEnabled': bool(self.motion and self.motion.armed and self.lease and self.heartbeat_at is not None),
                'motion': self.motion.status() if self.motion else None,
                'heartbeatTimeoutSeconds': self.HEARTBEAT_SECONDS,
                'lastEndReason': self.last_end_reason,
            }

    def assign(self, actor, owner, minutes):
        with self.lock:
            self._expire()
            if actor.get('role') != 'admin':
                raise SessionError('Administrator required', 403)
            if self.lease:
                raise SessionError('An exclusive session already exists')
            if type(minutes) is not int or not 1 <= minutes <= 60:
                raise SessionError('Duration must be 1–60 minutes', 400)
            if not isinstance(owner, dict) or not isinstance(owner.get('id'), str) or not owner['id'] or not isinstance(owner.get('name'), str):
                raise SessionError('A learner is required', 400)
            self.deadline = self.clock() + minutes * 60
            self.heartbeat_at = None
            self.disconnected = False
            self.last_end_reason = None
            self.lease = {
                'id': str(uuid.uuid4()), 'ownerId': owner['id'], 'ownerName': owner['name'][:100],
                'expiresAt': int((self.wall_clock() + minutes * 60) * 1000),
            }
            return self.snapshot()

    def update(self, actor, session_id, action, data=None):
        with self.lock:
            self._expire()
            if action == 'local':
                if actor.get('role') != 'admin':
                    raise SessionError('Administrator required', 403)
                if self.lease:
                    raise SessionError('End the current session first')
                if self.motion:
                    self.motion.return_local()
                return self.snapshot()
            if not self.lease or self.lease['id'] != session_id:
                raise SessionError('Session has ended or changed')
            owner = actor.get('id') == self.lease['ownerId']
            if action == 'heartbeat':
                if not owner:
                    raise SessionError('Only the assigned learner can join', 403)
                self.heartbeat_at = self.clock()
                self.disconnected = False
            elif action in ('release', 'stop'):
                if not owner and actor.get('role') != 'admin':
                    raise SessionError('Session owner or administrator required', 403)
                self._end('stopped' if action == 'stop' else 'revoked' if actor.get('role') == 'admin' else 'released')
            elif action == 'arm':
                if actor.get('role') != 'admin':
                    raise SessionError('Administrator required', 403)
                if not self.motion or not data or data.get('workspaceClear') is not True:
                    raise SessionError('Confirm the workspace is clear and supervised', 400)
                try:
                    self.motion.arm()
                except SessionError:
                    raise
                except Exception as error:
                    raise SessionError('Cannot read servo configuration; check the robot connection', 503) from error
            elif action in ('connect', 'start'):
                if actor.get('role') != 'admin':
                    raise SessionError('Administrator required', 403)
                if not self.motion:
                    raise SessionError('Movement unavailable', 423)
                if action == 'start' and (not data or data.get('workspaceClear') is not True):
                    raise SessionError('Confirm the workspace is clear and supervised', 400)
                try:
                    self.motion.connect() if action == 'connect' else self.motion.start_pose()
                except SessionError:
                    raise
                except Exception as error:
                    raise SessionError('Cannot reach the arm; check the robot connection', 503) from error
            elif action == 'move':
                if not owner or self.heartbeat_at is None:
                    raise SessionError('Only the joined learner can move the robot', 403)
                if not self.motion:
                    raise SessionError('Movement unavailable', 423)
                try:
                    self.motion.move((data or {}).get('joint'), (data or {}).get('delta'), (data or {}).get('target'))
                except SessionError as error:
                    if error.status == 503:
                        self.motion.stop()
                        self.motion.fault = error.message
                    raise
                except Exception as error:
                    self.motion.stop()
                    self.motion.fault = 'Servo communication failed; check the physical robot'
                    raise SessionError(self.motion.fault, 503) from error
            else:
                raise SessionError('Unsupported action', 400)
            return self.snapshot()

    def watchdog(self):
        while True:
            self.snapshot()
            time.sleep(0.25)


def session_blueprint(sessions, token):
    api = Blueprint('lab_sessions', __name__)

    @api.route('/api/lab/session', methods=['GET', 'POST'])
    def handle_session():
        # Fail closed when the shared server credential has not been provisioned.
        supplied = request.headers.get('Authorization', '')
        if not token or not hmac.compare_digest(supplied.encode('utf-8'), ('Bearer ' + token).encode('utf-8')):
            return jsonify({'error': 'Controller authentication required'}), 401
        if request.method == 'GET':
            return jsonify(sessions.snapshot()), 200, {'Cache-Control': 'no-store'}
        if request.content_length is None or request.content_length > 4096:
            return jsonify({'error': 'Invalid request size'}), 400
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get('actor'), dict):
            return jsonify({'error': 'Invalid request'}), 400
        try:
            if data.get('action') == 'assign':
                result = sessions.assign(data['actor'], data.get('owner'), data.get('minutes'))
            else:
                result = sessions.update(data['actor'], data.get('sessionId'), data.get('action'), data)
            return jsonify(result), 200, {'Cache-Control': 'no-store'}
        except SessionError as error:
            return jsonify({'error': error.message}), error.status

    return api
