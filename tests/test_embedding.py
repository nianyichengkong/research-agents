"""Embedding module tests -- behavior: creates Zhipu Embedding-3 client."""

from unittest.mock import MagicMock, patch


class TestCreateEmbedding:
    """create_embedding() returns an OpenAI-compatible embeddings client."""

    @patch("src.embedding.OpenAIEmbeddings")
    def test_initializes_with_zhipu_embedding_params(self, mock_embeddings):
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance

        from src.embedding import create_embedding

        result = create_embedding("test-key")

        mock_embeddings.assert_called_once_with(
            api_key="test-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model="embedding-3",
        )
        assert result == mock_instance
