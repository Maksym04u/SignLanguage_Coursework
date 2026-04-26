"""initial tables

Revision ID: 20260426_0001
Revises:
Create Date: 2026-04-26 12:45:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "20260426_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "users" not in existing_tables:
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("email", sa.String(), nullable=False),
            sa.Column("password_hash", sa.String(), nullable=False),
            sa.Column("preferred_language", sa.String(), nullable=False, server_default="en"),
        )
        op.create_index("ix_users_email", "users", ["email"], unique=True)
        op.create_index("ix_users_id", "users", ["id"], unique=False)

    if "translation_history" not in existing_tables:
        op.create_table(
            "translation_history",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("source_language", sa.String(), nullable=False),
            sa.Column("raw_text", sa.Text(), nullable=False),
            sa.Column("corrected_text", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        )
        op.create_index("ix_translation_history_id", "translation_history", ["id"], unique=False)
        op.create_index("ix_translation_history_user_id", "translation_history", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_translation_history_user_id", table_name="translation_history")
    op.drop_index("ix_translation_history_id", table_name="translation_history")
    op.drop_table("translation_history")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
