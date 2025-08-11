from contextvars import ContextVar

current_api_key: ContextVar[str] = ContextVar("current_api_key")