from langchain_openai import ChatOpenAI


def create_llm(api_key: str, base_url: str, model: str, tools: list | None = None) -> ChatOpenAI:
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )
    if tools:
        return llm.bind_tools(tools)
    return llm
