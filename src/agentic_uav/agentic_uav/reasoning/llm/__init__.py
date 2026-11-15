from agentic_uav.reasoning.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolSpec

# Names accepted for the OpenAI-compatible adapter (local Ollama / vLLM servers).
OPENAI_COMPAT = ("openai_compat", "ollama", "vllm")


def make_provider(name: str, model: str = "", api_key: str = "", requests_per_minute: float = 0.0,
                  log=None, thinking_level: str = "", timeout_s: float = 90.0,
                  base_url: str = "", temperature=None, reasoning_effort: str = "") -> LLMProvider:
    """Build a provider by name. Add new adapters here."""
    if name in OPENAI_COMPAT:
        from agentic_uav.reasoning.llm.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(model=model, base_url=base_url or "http://127.0.0.1:11434/v1",
                                    api_key=api_key, temperature=temperature, timeout_s=timeout_s,
                                    reasoning_effort=reasoning_effort)
    if name == "gemini":
        from agentic_uav.reasoning.llm.gemini import GeminiProvider
        return GeminiProvider(model=model, api_key=api_key, requests_per_minute=requests_per_minute, log=log,
                              thinking_level=thinking_level, timeout_s=timeout_s)
    raise ValueError(f"Unknown LLM provider '{name}'.")


__all__ = ["LLMProvider", "LLMResponse", "Message", "ToolCall", "ToolSpec", "make_provider"]
