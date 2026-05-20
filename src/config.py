import os
from dataclasses import dataclass


@dataclass
class Config:
    zhipu_api_key: str
    tavily_api_key: str
    langchain_api_key: str = ""
    langchain_project: str = "research-agent"
    max_iterations: int = 15
    max_searches: int = 8
    timeout_seconds: int = 600
    chroma_db_path: str = "./chroma_db"

    @classmethod
    def load(cls) -> "Config":
        zhipu_api_key = os.environ.get("ZHIPU_API_KEY")
        if not zhipu_api_key:
            raise ValueError("ZHIPU_API_KEY is required")

        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required")

        return cls(
            zhipu_api_key=zhipu_api_key,
            tavily_api_key=tavily_api_key,
            langchain_api_key=os.environ.get("LANGCHAIN_API_KEY", ""),
            langchain_project=os.environ.get("LANGCHAIN_PROJECT", "research-agent"),
            max_iterations=int(os.environ.get("MAX_ITERATIONS", "15")),
            max_searches=int(os.environ.get("MAX_SEARCHES", "8")),
            timeout_seconds=int(os.environ.get("TIMEOUT_SECONDS", "600")),
            chroma_db_path=os.environ.get("CHROMA_DB_PATH", "./chroma_db"),
        )
