"""Config module tests — behavior: loads API keys from .env and provides safety bounds."""

import os
import pytest
from unittest.mock import patch


class TestConfigLoad:
    """Config.load() reads env vars and returns a config object."""

    def test_loads_api_keys_from_env(self):
        with patch.dict(os.environ, {
            "ZHIPU_API_KEY": "test-zhipu-key",
            "TAVILY_API_KEY": "test-tavily-key",
            "LANGCHAIN_API_KEY": "test-langchain-key",
            "LANGCHAIN_PROJECT": "test-project",
        }):
            from src.config import Config
            config = Config.load()

            assert config.zhipu_api_key == "test-zhipu-key"
            assert config.tavily_api_key == "test-tavily-key"
            assert config.langchain_api_key == "test-langchain-key"
            assert config.langchain_project == "test-project"

    def test_provides_default_safety_bounds(self):
        with patch.dict(os.environ, {
            "ZHIPU_API_KEY": "key",
            "TAVILY_API_KEY": "key",
        }, clear=False):
            from src.config import Config
            config = Config.load()

            assert config.max_iterations == 15
            assert config.max_searches == 8
            assert config.timeout_seconds == 600
            assert config.chroma_db_path == "./chroma_db"

    def test_safety_bounds_override_from_env(self):
        with patch.dict(os.environ, {
            "ZHIPU_API_KEY": "key",
            "TAVILY_API_KEY": "key",
            "MAX_ITERATIONS": "5",
            "MAX_SEARCHES": "3",
            "TIMEOUT_SECONDS": "120",
        }, clear=False):
            from src.config import Config
            config = Config.load()

            assert config.max_iterations == 5
            assert config.max_searches == 3
            assert config.timeout_seconds == 120

    def test_chroma_db_path_override_from_env(self):
        with patch.dict(os.environ, {
            "ZHIPU_API_KEY": "key",
            "TAVILY_API_KEY": "key",
            "CHROMA_DB_PATH": "/tmp/test-chroma",
        }, clear=False):
            from src.config import Config
            config = Config.load()

            assert config.chroma_db_path == "/tmp/test-chroma"

    def test_raises_when_required_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.config import Config
            with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
                Config.load()
