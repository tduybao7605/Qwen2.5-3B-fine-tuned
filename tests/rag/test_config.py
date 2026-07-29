import pytest
from rag import config


def test_embedding_model_locked():
    assert config.EMBEDDING_MODEL == "bge-m3"
    assert config.EMBEDDING_DIM == 1024


def test_chunk_separator_is_markdown_hr():
    assert config.CHUNK_SEPARATOR == "\n---\n"


def test_kb_paths_exist():
    assert config.KB_DIR.is_dir()
    assert config.DB_FILE.is_file()


def test_threshold_in_valid_range():
    assert 0.0 < config.SCORE_THRESHOLD < 1.0


def test_dify_base_url_has_v1_suffix():
    assert config.DIFY_BASE_URL.endswith("/v1")
