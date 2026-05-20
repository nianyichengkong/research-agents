from langchain_openai import OpenAIEmbeddings


def create_embedding(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="embedding-3",
    )
