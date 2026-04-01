"""Unified LLM client for multiple providers (Anthropic, OpenAI, OpenRouter)."""

import os
from typing import Dict, Any, Tuple
from anthropic import Anthropic
from openai import OpenAI


def create_llm_client(model: str) -> Tuple[Any, str]:
    """
    Create appropriate API client based on model name.

    Args:
        model: Model name (e.g., "gpt-5.2", "google/gemini-3-pro-preview", "claude-sonnet-4-5-20250929")

    Returns:
        Tuple of (client, provider_name)

    Raises:
        ValueError: If required API key is missing
    """
    if model.startswith("gpt-"):
        # OpenAI direct API (GPT-4, GPT-5 series)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for GPT models")
        return OpenAI(api_key=api_key, timeout=120.0), "openai"

    elif "/" in model:
        # OpenRouter format - determine specific provider from model prefix
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter models")

        # Map model prefix to provider name for display
        if model.startswith("google/"):
            provider_name = "google"
        elif model.startswith("deepseek/"):
            provider_name = "deepseek"
        elif model.startswith("openai/"):
            provider_name = "openai"
        elif model.startswith("anthropic/"):
            provider_name = "anthropic"
        else:
            provider_name = "openrouter"

        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=120.0), provider_name

    else:
        # Default to Anthropic (claude-*)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Claude models")
        return Anthropic(api_key=api_key), "anthropic"


def call_llm(
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float = 1.0,
    system: str = None
) -> Dict[str, Any]:
    """
    Unified interface for calling any LLM provider.

    Args:
        model: Model name
        messages: List of message dicts with 'role' and 'content' keys
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 to 2.0)
        system: Optional system prompt (for Anthropic API)

    Returns:
        Dict with:
            - 'content': Response text
            - 'usage': Dict with 'input_tokens' and 'output_tokens'
            - 'provider': Provider name ('anthropic', 'openai', 'google', 'deepseek', 'openrouter')

    Raises:
        ValueError: If API key is missing or API call fails
    """
    client, provider = create_llm_client(model)

    if provider == "anthropic":
        # Anthropic API
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            "provider": provider
        }

    else:
        # OpenAI/OpenRouter API
        # Convert system prompt to OpenAI format if provided
        api_messages = messages.copy()
        if system:
            api_messages = [{"role": "system", "content": system}] + api_messages

        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            max_completion_tokens=max_tokens,  # OpenAI uses max_completion_tokens instead of max_tokens
            temperature=temperature
        )

        # Extract content from response
        content = response.choices[0].message.content

        # Handle None or empty content (safety filters, refusals, etc.)
        if content is None:
            # Check for refusal
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                content = f"[REFUSAL] {response.choices[0].message.refusal}"
            # Check finish reason
            elif hasattr(response.choices[0], 'finish_reason'):
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'content_filter':
                    content = "[CONTENT_FILTER] Response blocked by safety filter"
                elif finish_reason == 'length':
                    content = "[LENGTH] Response truncated at max_completion_tokens"
                else:
                    content = f"[EMPTY_RESPONSE] finish_reason={finish_reason}"
            else:
                content = "[EMPTY_RESPONSE] No content returned"

        return {
            "content": content,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            },
            "provider": provider
        }
