"""
Utility functions for user authentication and chat persistence.

This module encapsulates all database interactions and helper functions
related to user and conversation management. It uses SQLite for
persistence and stores conversations, messages and IP addresses locally.

To use these functions in your Streamlit app, import them as needed:

```
from auth_db import (
    init_db,
    get_or_create_user,
    create_conversation,
    get_conversations,
    get_messages,
    save_message,
    update_conversation_title,
    delete_conversation,
)
```
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional
import urllib.request

# Define the path to the SQLite database. It will reside alongside this file.
DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")


def init_db() -> None:
    """
    Initialize the SQLite database with the required tables. This function
    creates tables for users, conversations, and messages if they do not
    already exist. It also adds an `ip_address` column to the conversations
    table for legacy databases.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # users table
    cur.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "email TEXT UNIQUE, "
        "name TEXT)"
    )
    # conversations table with optional IP column
    cur.execute(
        "CREATE TABLE IF NOT EXISTS conversations ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "user_id INTEGER, "
        "title TEXT, "
        "created_at TEXT, "
        "ip_address TEXT, "
        "FOREIGN KEY(user_id) REFERENCES users(id)"
        ")"
    )
    # messages table
    cur.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "conversation_id INTEGER, "
        "role TEXT, "
        "content TEXT, "
        "created_at TEXT, "
        "FOREIGN KEY(conversation_id) REFERENCES conversations(id)"
        ")"
    )
    # attempt to add ip_address column if missing
    try:
        cur.execute("ALTER TABLE conversations ADD COLUMN ip_address TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()


def get_client_ip() -> str:
    """
    Attempt to retrieve the client's IP address from environment variables or
    a fallback service. Returns 'unknown' if it cannot be determined.
    """
    ip = os.environ.get("REMOTE_ADDR") or os.environ.get("HTTP_X_FORWARDED_FOR") or "unknown"
    if ip == "unknown":
        try:
            ip = urllib.request.urlopen("https://api.ipify.org").read().decode("utf8")
        except Exception:
            ip = "unknown"
    return ip


def get_or_create_user(email: str, name: str) -> int:
    """
    Return the user ID for a given email/name, creating a new user if necessary.
    Uses the email as the primary key. If no email is provided, the name is used
    instead to ensure uniqueness.
    """
    if not email:
        email = name
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    if row:
        uid = row[0]
    else:
        cur.execute("INSERT INTO users (email, name) VALUES (?, ?)", (email, name))
        conn.commit()
        uid = cur.lastrowid
    conn.close()
    return uid


def create_conversation(user_id: int, title: Optional[str] = None, ip_address: Optional[str] = None) -> int:
    """
    Create a new conversation row and return its ID. If no `ip_address` is
    provided, the client's IP address is detected automatically.
    """
    if ip_address is None:
        ip_address = get_client_ip()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO conversations (user_id, title, created_at, ip_address) VALUES (?, ?, ?, ?)",
        (user_id, title, now, ip_address),
    )
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return conv_id


def get_conversations(user_id: int):
    """
    Return a list of (id, title, created_at) tuples for the user's conversations,
    sorted by most recent first.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_messages(conv_id: int):
    """
    Return a list of message dictionaries for the given conversation ID, ordered
    by insertion.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": role, "message": content} for role, content in rows]


def save_message(conv_id: int, role: str, content: str) -> None:
    """
    Persist a chat message under the given conversation. No action is taken if
    `conv_id` is None.
    """
    if conv_id is None:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, now),
    )
    conn.commit()
    conn.close()


def update_conversation_title(conv_id: int, new_title: str) -> None:
    """
    Update the title of an existing conversation. If `new_title` is empty,
    nothing is changed.
    """
    if not new_title:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, conv_id))
    conn.commit()
    conn.close()


def delete_conversation(conv_id: int) -> None:
    """
    Remove a conversation and all its messages from the database.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    cur.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()