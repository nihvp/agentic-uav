from agentic_uav.reasoning.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec


def make_provider(name: str, model: str = "", api_key: str = "") -> LLMProvider:
    """Build a provider by name. Add new adapters here."""
    if name == "gemini":
        from agentic_uav.reasoning.llm.gemini import GeminiProvider
        return GeminiProvider(model=model, api_key=api_key)
    raise ValueError(f"Unknown LLM provider '{name}'.")


__all__ = ["LLMProvider", "LLMResponse", "Message", "ToolCall", "ToolSpec", "make_provider"]
