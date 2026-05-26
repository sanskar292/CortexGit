import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")

# For SQLite, NullPool closes connections too aggressively for in-memory databases to persist schema.
# Standard connection pooling (None) is preferred.
pool_class = NullPool if "sqlite" not in TEST_DATABASE_URL else None

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=pool_class)
TestingSessionLocal = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
