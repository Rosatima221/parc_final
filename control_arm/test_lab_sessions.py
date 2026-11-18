"""Run with: /usr/bin/python3 -m unittest test_lab_sessions -v"""
import threading
import unittest
from flask import Flask
from lab_sessions import LabSessions, SessionError, session_blueprint


ADMIN = {'id': 'admin', 'role': 'admin'}
OWNER = {'id': 'learner', 'role': 'learner', 'name': 'Learner'}
OTHER = {'id': 'other', 'role': 'learner', 'name': 'Other'}


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.now = 100
        self.sessions = LabSessions(clock=lambda: self.now, wall_clock=lambda: 1000)

    def assign(self, minutes=1):
        return self.sessions.assign(ADMIN, OWNER, minutes)['session']['id']

    def test_concurrent_assignment_has_one_winner(self):
        barrier = threading.Barrier(3)
        outcomes = []
        def assign():
            barrier.wait()
            try:
                self.assign()
                outcomes.append('assigned')
            except SessionError:
                outcomes.append('conflict')
        threads = [threading.Thread(target=assign) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertCountEqual(outcomes, ['assigned', 'conflict'])

    def test_permissions(self):
        with self.assertRaises(SessionError):
            self.sessions.assign(OTHER, OWNER, 1)
        session_id = self.assign()
        for action in ('heartbeat', 'release'):
            with self.assertRaises(SessionError):
                self.sessions.update(OTHER, session_id, action)
        with self.assertRaises(SessionError):
            self.sessions.update(ADMIN, session_id, 'heartbeat')
        self.sessions.update(ADMIN, session_id, 'release')
        self.assertIsNone(self.sessions.snapshot()['session'])

    def test_owner_release(self):
        self.sessions.update(OWNER, self.assign(), 'release')
        self.assertEqual(self.sessions.snapshot()['lastEndReason'], 'released')

    def test_quiet_page_disconnects_but_keeps_the_slot_and_can_rejoin(self):
        session_id = self.assign()
        self.sessions.update(OWNER, session_id, 'heartbeat')
        self.now += 9
        state = self.sessions.snapshot()
        self.assertTrue(state['session']['joined'])
        self.assertFalse(state['session']['disconnected'])
        self.now += 1
        state = self.sessions.snapshot()
        self.assertEqual(state['session']['id'], session_id)
        self.assertFalse(state['session']['joined'])
        self.assertTrue(state['session']['disconnected'])
        self.assertIsNone(state['lastEndReason'])
        self.sessions.update(OWNER, session_id, 'heartbeat')
        state = self.sessions.snapshot()
        self.assertTrue(state['session']['joined'])
        self.assertFalse(state['session']['disconnected'])

    def test_disconnected_session_still_expires_and_can_be_revoked(self):
        session_id = self.assign()
        self.sessions.update(OWNER, session_id, 'heartbeat')
        self.now += 10
        self.assertTrue(self.sessions.snapshot()['session']['disconnected'])
        self.now += 60
        self.assertIsNone(self.sessions.snapshot()['session'])
        self.assertEqual(self.sessions.snapshot()['lastEndReason'], 'expired')
        session_id = self.assign(5)
        self.sessions.update(OWNER, session_id, 'heartbeat')
        self.now += 10
        self.sessions.update(ADMIN, session_id, 'release')
        self.assertIsNone(self.sessions.snapshot()['session'])
        self.assertEqual(self.sessions.snapshot()['lastEndReason'], 'revoked')

    def test_expiry_applies_even_with_heartbeats(self):
        session_id = self.assign()
        for _ in range(11):
            self.now += 5
            self.sessions.update(OWNER, session_id, 'heartbeat')
        self.now += 5
        with self.assertRaises(SessionError):
            self.sessions.update(OWNER, session_id, 'heartbeat')
        self.assertEqual(self.sessions.snapshot()['lastEndReason'], 'expired')

    def test_old_browser_cannot_update_new_session(self):
        old_id = self.assign()
        self.sessions.update(ADMIN, old_id, 'release')
        new_id = self.assign()
        self.assertNotEqual(old_id, new_id)
        for action in ('heartbeat', 'release'):
            with self.assertRaises(SessionError):
                self.sessions.update(OWNER, old_id, action)
        self.assertEqual(self.sessions.snapshot()['session']['id'], new_id)

    def test_invalid_durations(self):
        for value in (True, 0, 61, 1.5, '5', None):
            with self.assertRaises(SessionError):
                self.assign(value)

    def test_physical_control_remains_disabled(self):
        self.assign()
        self.assertFalse(self.sessions.snapshot()['controlEnabled'])

    def test_api_requires_server_credential(self):
        app = Flask(__name__)
        app.register_blueprint(session_blueprint(self.sessions, 'secret'))
        client = app.test_client()
        self.assertEqual(client.get('/api/lab/session').status_code, 401)
        self.assertEqual(client.post('/api/lab/session', json={}).status_code, 401)
        headers = {'Authorization': 'Bearer secret'}
        self.assertEqual(client.get('/api/lab/session', headers=headers).status_code, 200)
        self.assertEqual(client.post('/api/lab/session', headers=headers, json=[]).status_code, 400)
        assigned = client.post('/api/lab/session', headers=headers, json={
            'action': 'assign', 'actor': ADMIN, 'owner': OWNER, 'minutes': 1,
        })
        self.assertEqual(assigned.status_code, 200)
        self.assertFalse(assigned.json['controlEnabled'])

    def test_unconfigured_credential_fails_closed(self):
        app = Flask(__name__)
        app.register_blueprint(session_blueprint(self.sessions, ''))
        response = app.test_client().get('/api/lab/session', headers={'Authorization': 'Bearer '})
        self.assertEqual(response.status_code, 401)


if __name__ == '__main__':
    unittest.main()
