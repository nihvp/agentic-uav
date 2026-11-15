from agentic_uav.reasoning.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec


def make_provider(name: str, model: str = "", api_key: str = "", requests_per_minute: float = 0.0,
                  log=None, thinking_level: str = "", timeout_s: float = 90.0) -> LLMProvider:
    """Build a provider by name. Add new adapters here."""
    if name == "gemini":
        from agentic_uav.reasoning.llm.gemini import GeminiProvider
        return GeminiProvider(model=model, api_key=api_key, requests_per_minute=requests_per_minute, log=log,
                              thinking_level=thinking_level, timeout_s=timeout_s)
    raise ValueError(f"Unknown LLM provider '{name}'.")


__all__ = ["LLMProvider", "LLMResponse", "Message", "ToolCall", "ToolSpec", "make_provider"]
