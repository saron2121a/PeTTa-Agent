
import json
import sqlite3

from petta import PeTTa


class AgentMemory:

    VALID_MEMORY_TYPES = {
        "context",
        "skill",
        "episodic",
        "semantic",
        "procedural"
    }

    MEMORY_SPACE = "&agent_memories"

    def __init__(self, db_path="agent_memory_store.db"):
        self.db_path = db_path
        self.petta = PeTTa()
        self.create_table()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def create_table(self):
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(memory_type, subject, relation, object)
                )
            """)

    def save_memory(self, memory_type, subject, relation, obj):
        if memory_type not in self.VALID_MEMORY_TYPES:
            raise ValueError(f"Invalid memory type: {memory_type}")

        values = (subject, relation, obj)

        if not all(
            isinstance(value, str) and value.strip()
            for value in values
        ):
            raise ValueError("Memory values must be non-empty strings")

        with self.connect() as connection:
            cursor = connection.execute("""
                INSERT OR IGNORE INTO memories
                (memory_type, subject, relation, object)
                VALUES (?, ?, ?, ?)
            """, (memory_type, subject, relation, obj))

        if cursor.rowcount:
            self._add_to_petta(memory_type, subject, relation, obj)

        return bool(cursor.rowcount)

    def _add_to_petta(self, memory_type, subject, relation, obj):
        atom = (
            f"(agent-memory "
            f"{json.dumps(memory_type, ensure_ascii=False)} "
            f"{json.dumps(subject, ensure_ascii=False)} "
            f"{json.dumps(relation, ensure_ascii=False)} "
            f"{json.dumps(obj, ensure_ascii=False)})"
        )

        code = f"!(add-atom {self.MEMORY_SPACE} {atom})"

        return self.petta.process_metta_string(code)

    def _clear_petta_memory(self):
        code = (
            f"!(collapse (match {self.MEMORY_SPACE} $ATOM "
            f"(remove-atom {self.MEMORY_SPACE} $ATOM)))"
        )

        return self.petta.process_metta_string(code)

    def restore_to_petta(self):
        with self.connect() as connection:
            memories = connection.execute("""
                SELECT memory_type, subject, relation, object
                FROM memories
                ORDER BY id
            """).fetchall()

        # Clear only the dedicated agent memory space.
        self._clear_petta_memory()

        # Rebuild the space from SQLite.
        for memory_type, subject, relation, obj in memories:
            self._add_to_petta(
                memory_type,
                subject,
                relation,
                obj
            )

        return len(memories)

    def query_memory(self, memory_type, subject):
        if memory_type not in self.VALID_MEMORY_TYPES:
            raise ValueError(f"Invalid memory type: {memory_type}")

        if not isinstance(subject, str):
            raise ValueError("Subject must be a string")

        code = (
            f"!(collapse (match {self.MEMORY_SPACE} "
            f"(agent-memory {json.dumps(memory_type, ensure_ascii=False)} "
            f"{json.dumps(subject, ensure_ascii=False)} $R $O) "
            f"($R $O)))"
        )

        return self.petta.process_metta_string(code)

    def get_relevant_memories(self, task, limit=5):
        import re

        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a non-empty string")

        if type(limit) is not int or limit < 1:
            raise ValueError("Limit must be a positive integer")

        stop_words = {
            "the", "and", "for", "with", "from",
            "that", "this", "what", "how", "can",
            "you", "are", "use", "using", "continue"
        }

        words = set(re.findall(r"[a-zA-Z0-9_'-]{3,}", task.lower()))
        keywords = sorted(words - stop_words)

        if not keywords:
            return []

        conditions = []
        parameters = []

        for word in keywords:
            conditions.extend([
                "LOWER(subject) LIKE ?",
                "LOWER(relation) LIKE ?",
                "LOWER(object) LIKE ?"
            ])
            parameters.extend([f"%{word}%"] * 3)

        sql = f"""
            SELECT id, memory_type, subject, relation, object
            FROM memories
            WHERE {" OR ".join(conditions)}
        """

        with self.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()

        def relevance_score(row):
            content = " ".join(str(value).lower() for value in row[1:])
            return sum(1 for word in keywords if word in content)

        rows.sort(key=relevance_score, reverse=True)

        return [
            {
                "id": row[0],
                "memory_type": row[1],
                "subject": row[2],
                "relation": row[3],
                "object": row[4]
            }
            for row in rows[:limit]
        ]

    def search_memories(self, keyword, memory_type=None):
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Keyword must be a non-empty string")

        if (
            memory_type is not None
            and memory_type not in self.VALID_MEMORY_TYPES
        ):
            raise ValueError(f"Invalid memory type: {memory_type}")

        pattern = f"%{keyword}%"

        with self.connect() as connection:
            if memory_type:
                rows = connection.execute("""
                    SELECT id, memory_type, subject, relation, object
                    FROM memories
                    WHERE memory_type = ?
                    AND (
                        subject LIKE ?
                        OR relation LIKE ?
                        OR object LIKE ?
                    )
                    ORDER BY id
                """, (
                    memory_type,
                    pattern,
                    pattern,
                    pattern
                )).fetchall()
            else:
                rows = connection.execute("""
                    SELECT id, memory_type, subject, relation, object
                    FROM memories
                    WHERE subject LIKE ?
                    OR relation LIKE ?
                    OR object LIKE ?
                    ORDER BY id
                """, (
                    pattern,
                    pattern,
                    pattern
                )).fetchall()

        return [
            {
                "id": row[0],
                "memory_type": row[1],
                "subject": row[2],
                "relation": row[3],
                "object": row[4]
            }
            for row in rows
        ]
