from grounded_rag.db.models import Base


def test_metadata_declares_the_expected_tables() -> None:
    assert set(Base.metadata.tables) == {"documents", "chunks", "api_keys"}


def test_search_vector_is_a_stored_generated_column() -> None:
    column = Base.metadata.tables["chunks"].c.search_vector
    assert column.computed is not None
    assert column.computed.persisted is True
    assert "context_header" in str(column.computed.sqltext)


def test_chunks_are_removed_with_their_document() -> None:
    foreign_key = next(iter(Base.metadata.tables["chunks"].c.doc_id.foreign_keys))
    assert foreign_key.ondelete == "CASCADE"
