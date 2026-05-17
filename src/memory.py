from langgraph.checkpoint.sqlite import SqliteSaver


def create_checkpointer(db_path: str = "research_agent.db"):
    """Return a context manager that yields a SqliteSaver checkpointer.

    Usage:
        with create_checkpointer("path.db") as checkpointer:
            graph = create_agent_graph(..., checkpointer=checkpointer)
    """
    return SqliteSaver.from_conn_string(db_path)
