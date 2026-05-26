"""enable_pgvector

Revision ID: bfc3db3b55cf
Revises: afcedfacf8d9
Create Date: 2026-05-24 18:04:56.306384

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bfc3db3b55cf'
down_revision: Union[str, Sequence[str], None] = 'afcedfacf8d9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        # SQLite doesn't support pgvector, do nothing
        return

    # Check if vector extension is in pg_available_extensions
    try:
        res = conn.execute(sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")).fetchone()
        if res is not None:
            op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            op.execute("ALTER TABLE snapshot_store ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector;")
    except Exception:
        pass


def downgrade() -> None:
    """Downgrade schema."""
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        # SQLite doesn't support pgvector, do nothing
        return

    try:
        res = conn.execute(sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")).fetchone()
        if res is not None:
            op.execute("ALTER TABLE snapshot_store ALTER COLUMN embedding TYPE double precision[] USING embedding::double precision[];")
    except Exception:
        pass
