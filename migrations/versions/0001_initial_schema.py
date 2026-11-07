"""Documents, chunks and API keys.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

SEARCH_VECTOR = "to_tsvector('english', coalesce(context_header, '') || ' ' || text)"


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("corpus", sa.String(64), nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("allowed_roles", postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("char_count", sa.Integer, nullable=False),
        sa.Column("chunk_count", sa.Integer, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status in ('pending', 'ready', 'deleting')", name="ck_documents_status"
        ),
    )
    op.create_index("ix_documents_corpus_status", "documents", ["corpus", "status"])
    op.create_index("ix_documents_corpus_source", "documents", ["corpus", "source"])
    op.create_index(
        "ix_documents_allowed_roles", "documents", ["allowed_roles"], postgresql_using="gin"
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column(
            "doc_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("start_char", sa.Integer, nullable=False),
        sa.Column("end_char", sa.Integer, nullable=False),
        sa.Column("section", sa.Text, nullable=True),
        sa.Column("context_header", sa.Text, nullable=False),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR,
            sa.Computed(SEARCH_VECTOR, persisted=True),
            nullable=False,
        ),
        sa.UniqueConstraint("doc_id", "ordinal", name="uq_chunks_doc_ordinal"),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_search_vector", "chunks", ["search_vector"], postgresql_using="gin")

    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("key_prefix", sa.String(16), nullable=False),
        sa.Column("roles", postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("api_keys")
    op.drop_table("chunks")
    op.drop_table("documents")
