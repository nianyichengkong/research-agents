import os

from langchain_core.tools import tool


def create_search_tools(tavily_api_key: str) -> list:
    """Create search and extraction tools with the given API key."""
    from langchain_tavily import TavilySearch

    tavily_search = TavilySearch(
        max_results=5,
        tavily_api_key=tavily_api_key,
    )

    api_key = tavily_api_key

    @tool
    def web_extract(urls: list[str]) -> str:
        """Extract main content from web pages. Provide a list of URLs to extract text from."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            results = client.extract(urls=urls)
            parts = []
            for r in results.get("results", []):
                parts.append(f"[{r.get('url', '')}]\n{r.get('text', '')}")
            return "\n\n---\n\n".join(parts) if parts else "No content extracted"
        except Exception as e:
            return f"Extraction error: {e}"

    return [tavily_search, web_extract]
