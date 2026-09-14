import sqlite3
from contextlib import contextmanager
from typing import Iterator

from src.app.config import DB_PATH


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                old_price REAL NOT NULL,
                suggested_price REAL NOT NULL,
                final_price REAL NOT NULL,
                run_timestamp TEXT NOT NULL
            )
            """
        )


def insert_price_history(rows: list[tuple[str, float, float, float, str]]) -> None:
    if not rows:
        return

    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO price_history (
                product_id,
                old_price,
                suggested_price,
                final_price,
                run_timestamp
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def fetch_price_history() -> list[dict]:
    with get_connection() as connection:
        result = connection.execute(
            """
            SELECT product_id, old_price, suggested_price, final_price, run_timestamp
            FROM price_history
            ORDER BY id DESC
            """
        )
        return [dict(row) for row in result.fetchall()]
