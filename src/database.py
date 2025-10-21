import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

DB_PATH = Path("./chat_sessions.db")

def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            document_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            time_taken REAL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    conn.commit()
    conn.close()

# ==================================
# === CLEANUP ORPHANED EMBEDDINGS AT STARTUP ===
# ==================================
def cleanup_orphaned_embeddings_on_startup():
    """
    Run orphaned data cleanup on app startup.
    This removes embeddings, images, AND documents for sessions 
    that no longer exist in the database.
    """
    try:
        from src.chatbot import cleanup_orphaned_data  # Updated function name
        import sqlite3
        
        # Get all active session IDs from database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM sessions")
        active_sessions = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"\n Checking for orphaned data...")
        print(f" Active sessions in database: {len(active_sessions)}")
        
        # Run cleanup (now cleans embeddings, images, AND documents)
        stats = cleanup_orphaned_data(active_sessions)
        
        total_deleted = stats['embeddings_deleted'] + stats['images_deleted'] + stats['documents_deleted']
        if total_deleted > 0:
            print(f" Orphaned data cleanup complete")
        else:
            print(f" No orphaned data found")
    
    except Exception as e:
        print(f" Error during orphaned data cleanup: {e}")



class SessionManager:
    """Manages chat sessions with persistent storage."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
    
    def create_session(self, title: str = None, document_name: str = None) -> str:
        """Create a new session and return session ID."""
        session_id = f"session_{str(uuid.uuid4())[:8]}"
        timestamp = datetime.now().isoformat()
        
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sessions (id, title, document_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, title, document_name, timestamp, timestamp))
        
        conn.commit()
        conn.close()
        
        return session_id
 
    def load_session(self, session_id: str) -> dict:
        """Load a session by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        return {
            "id": result[0],
            "title": result[1],
            "document_name": result[2],
            "created_at": result[3],
            "updated_at": result[4]
        }
    
    def save_message(self, session_id: str, question: str, answer: str, elapsed_time: float):
        """Save Q&A pair to session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO messages (session_id, question, answer, time_taken, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, question, answer, elapsed_time, timestamp))
        
        # Update session updated_at
        cursor.execute("""
            UPDATE sessions SET updated_at = ? WHERE id = ?
        """, (timestamp, session_id))
        
        conn.commit()
        conn.close()
    
    def list_sessions(self) -> list:
        """List all sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.id, s.title, s.document_name, s.created_at, COUNT(m.id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": r[0],
                "title": r[1],
                "document_name": r[2] if r[2] else "Unknown",
                "created_at": r[3],
                "message_count": r[4] if r[4] else 0
            }
            for r in results
        ]
    
    def get_session_messages(self, session_id: str) -> list:
        """Get all messages from a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT question, answer, time_taken, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "question": r[0],
                "answer": r[1],
                "time_taken": r[2],
                "timestamp": r[3]
            }
            for r in results
        ]
    
    def update_document_name(self, session_id: str, document_name: str):
        """Update the document name for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions SET document_name = ?, updated_at = ?
            WHERE id = ?
        """, (document_name, datetime.now().isoformat(), session_id))
        
        conn.commit()
        conn.close()

    def get_conversation_history(self, session_id: str, max_entries: int = 3) -> str:
        """Get recent Q&A pairs as context string."""
        messages = self.get_session_messages(session_id)
        
        if not messages:
            return ""
        
        recent_messages = messages[-max_entries:]
        context = "=== PREVIOUS CONVERSATION HISTORY ===\n\n"
        for i, msg in enumerate(recent_messages, 1):
            question = msg["question"]
            answer = msg["answer"][:300] + "..." if len(msg["answer"]) > 300 else msg["answer"]
            context += f"Q{i}: {question}\nA{i}: {answer}\n\n"
        context += "=== END OF HISTORY ===\n\n"
        return context

    def delete_session(self, session_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    def clear_session_messages(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("""
            UPDATE sessions SET updated_at = ? WHERE id = ?
        """, (datetime.now().isoformat(), session_id))
        
        conn.commit()
        conn.close()
    
    def update_session_title(self, session_id: str, title: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions SET title = ?, updated_at = ?
            WHERE id = ?
        """, (title, datetime.now().isoformat(), session_id))
        
        conn.commit()
        conn.close()