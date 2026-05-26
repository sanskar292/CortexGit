import os
import sys
import pytest
from unittest.mock import MagicMock

# NOTE: Global monkey-patching of create_async_engine is fragile — prefer pytest fixtures + dependency injection.
# Intercept and patch create_async_engine and create_engine at the very beginning of the test session!
import sqlalchemy
import sqlalchemy.ext.asyncio
from sqlalchemy import event

original_create_async_engine = sqlalchemy.ext.asyncio.create_async_engine
original_create_engine = sqlalchemy.create_engine

def patched_create_async_engine(url, *args, **kwargs):
    env_url = os.getenv("TEST_DATABASE_URL")
    if env_url:
        target_url = env_url
    else:
        # Use a shared SQLite file so multiple connections/sessions see the same tables
        target_url = "sqlite+aiosqlite:///cortexgit_test.db"
    
    if "sqlite" in target_url:
        kwargs.pop("poolclass", None)
        connect_args = kwargs.setdefault("connect_args", {})
        connect_args["timeout"] = 60
    
    return original_create_async_engine(target_url, *args, **kwargs)

def patched_create_engine(url, *args, **kwargs):
    env_url = os.getenv("TEST_DATABASE_URL") or os.getenv("TEST_DB_URL_SYNC")
    if env_url:
        target_url = env_url
    else:
        # Sync SQLite pointing to the same shared test database file
        target_url = "sqlite:///cortexgit_test.db"
    
    if "sqlite" in target_url:
        connect_args = kwargs.setdefault("connect_args", {})
        connect_args["timeout"] = 60
    
    return original_create_engine(target_url, *args, **kwargs)

sqlalchemy.ext.asyncio.create_async_engine = patched_create_async_engine
sqlalchemy.create_engine = patched_create_engine

# Enable WAL mode and busy timeout on all SQLite connections to prevent locking issues
@event.listens_for(sqlalchemy.engine.Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    # Check if SQLite connection
    if dbapi_connection.__class__.__name__ == 'Connection' or 'sqlite' in str(type(dbapi_connection)):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA busy_timeout=60000;")
            cursor.close()
        except Exception:
            pass

# Mock psycopg2 connection calls to be no-ops during SQLite tests
try:
    import psycopg2
    original_psycopg2_connect = psycopg2.connect
    
    def patched_psycopg2_connect(*args, **kwargs):
        env_url = os.getenv("TEST_DATABASE_URL") or os.getenv("TEST_DB_URL_SYNC")
        if env_url and "postgresql" in env_url:
            return original_psycopg2_connect(*args, **kwargs)
        
        # SQLite mode -> return Mock psycopg2 connection
        mock_conn = MagicMock()
        return mock_conn
    
    psycopg2.connect = patched_psycopg2_connect
except ImportError:
    # If psycopg2 is not installed, mock it entirely so imports don't crash
    sys.modules["psycopg2"] = MagicMock()
    sys.modules["psycopg2.extensions"] = MagicMock()

def pytest_configure(config):
    # Ensure postgres marker is registered
    config.addinivalue_line(
        "markers", "postgres: mark test as requiring a running PostgreSQL database"
    )

@pytest.fixture(scope="session", autouse=True)
def cleanup_sqlite_db():
    # Setup: Remove existing test db files
    for db_file in ["cortexgit_test.db", "cortexgit_test.db-shm", "cortexgit_test.db-wal"]:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass
                
    yield
    
    # Teardown: Remove test db files after tests finish
    import time
    time.sleep(1.0)
    for db_file in ["cortexgit_test.db", "cortexgit_test.db-shm", "cortexgit_test.db-wal"]:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass
