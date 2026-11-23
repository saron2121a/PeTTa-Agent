
import json
import sqlite3


class CheckpointManager:

    def __init__(self, db_path="agent_checkpoints.db"):
        self.db_path = db_path
        self.create_table()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def create_table(self):
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    task_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    current_step INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_checkpoint(
        self,
        task_id,
        task,
        messages,
        current_step,
        status="running"
    ):
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Task ID must be a non-empty string")

        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")

        if type(current_step) is not int or current_step < 0:
            raise ValueError("Current step must be a non-negative integer")

        if status not in {"running", "completed", "failed"}:
            raise ValueError("Invalid checkpoint status")

        serialized_messages = json.dumps(messages)

        with self.connect() as connection:
            connection.execute("""
                INSERT INTO checkpoints
                (task_id, task, messages, current_step, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    task = excluded.task,
                    messages = excluded.messages,
                    current_step = excluded.current_step,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                task_id,
                task,
                serialized_messages,
                current_step,
                status
            ))


    def load_checkpoint(self, task_id):

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Task ID must be a non-empty string")

        with self.connect() as connection:
            row = connection.execute("""
                SELECT task_id, task, messages, current_step, status
                FROM checkpoints
                WHERE task_id = ?
            """, (task_id,)).fetchone()

        if row is None:
            return None

        return {
            "task_id": row[0],
            "task": row[1],
            "messages": json.loads(row[2]),
            "current_step": row[3],
            "status": row[4]
        }


    def delete_checkpoint(self, task_id):
        with self.connect() as connection:
            cursor = connection.execute("""
                DELETE FROM checkpoints
                WHERE task_id = ?
            """, (task_id,))

        return cursor.rowcount > 0
