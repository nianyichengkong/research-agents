"""LLM module tests — behavior: initializes ChatOpenAI with correct params and binds tools."""

from unittest.mock import patch, MagicMock
import pytest


class TestCreateLLM:
    """create_llm() returns a ChatOpenAI bound with tools."""

    @patch("src.llm.ChatOpenAI")
    def test_initializes_with_zhipu_params(self, mock_chat_openai):
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        from src.llm import create_llm

        result = create_llm(
            api_key="test-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model="glm-4",
        )

        mock_chat_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model="glm-4",
            temperature=0,
        )
        assert result == mock_instance

    @patch("src.llm.ChatOpenAI")
    def test_binds_tools_when_provided(self, mock_chat_openai):
        mock_instance = MagicMock()
        mock_bound = MagicMock()
        mock_instance.bind_tools.return_value = mock_bound
        mock_chat_openai.return_value = mock_instance

        from src.llm import create_llm

        tool = MagicMock()
        result = create_llm(
            api_key="test-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model="glm-4",
            tools=[tool],
        )

        mock_instance.bind_tools.assert_called_once_with([tool])
        assert result == mock_bound
