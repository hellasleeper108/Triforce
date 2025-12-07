import sqlite3
import json
import time
import os
from contextlib import contextmanager
from typing import List, Optional, Dict, Any

from triforce.odin.utils.logger import logger

DB_PATH = os.getenv("ODIN_DB_PATH", "odin.db")

class JobStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    worker_url TEXT,
                    job_type TEXT,
                    request_payload JSON,
                    result_payload JSON,
                    error TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    definition JSON,
                    state JSON,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.commit()
            logger.info(f"JobStore initialized at {self.db_path}")

    def add_job(self, job_id: str, job_type: str, payload: Dict[str, Any]):
        with self._get_conn() as conn:
            now = time.time()
            conn.execute("""
                INSERT INTO jobs (job_id, status, job_type, request_payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (job_id, "QUEUED", job_type, json.dumps(payload), now, now))
            conn.commit()

    def update_job(self, job_id: str, status: str, worker_url: Optional[str] = None, result: Optional[Dict] = None, error: Optional[str] = None):
        with self._get_conn() as conn:
            now = time.time()
            cursor = conn.cursor()
            
            updates = ["status = ?", "updated_at = ?"]
            params = [status, now]
            
            if worker_url is not None:
                updates.append("worker_url = ?")
                params.append(worker_url)
                
            if result is not None:
                updates.append("result_payload = ?")
                params.append(json.dumps(result))
                
            if error is not None:
                updates.append("error = ?")
                params.append(error)
                
            params.append(job_id)
            
            sql = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
            cursor.execute(sql, params)
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def get_recoverable_jobs(self) -> List[Dict[str, Any]]:
        """
        Returns a list of jobs that were in a running state and need to be queued again.
        This usually happens if the server crashed.
        """
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM jobs WHERE status = 'RUNNING'").fetchall()
            return [dict(row) for row in rows]

    def reset_job_to_queued(self, job_id: str):
        with self._get_conn() as conn:
            now = time.time()
            conn.execute("""
                UPDATE jobs 
                SET status = 'QUEUED', worker_url = NULL, updated_at = ? 
                WHERE job_id = ?
            """, (now, job_id))
            conn.commit()

    # --- Workflow Methods ---

    def add_workflow(self, workflow_id: str, definition_json: str, initial_state_json: str):
        with self._get_conn() as conn:
            now = time.time()
            conn.execute("""
                INSERT INTO workflows (workflow_id, status, definition, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (workflow_id, "RUNNING", definition_json, initial_state_json, now, now))
            conn.commit()

    def update_workflow_state(self, workflow_id: str, status: str, state_json: str):
        with self._get_conn() as conn:
            now = time.time()
            conn.execute("""
                UPDATE workflows 
                SET status = ?, state = ?, updated_at = ?
                WHERE workflow_id = ?
            """, (status, state_json, now, workflow_id))
            conn.commit()

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def get_active_workflows(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM workflows WHERE status = 'RUNNING'").fetchall()
            return [dict(row) for row in rows]
