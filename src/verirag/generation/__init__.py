"""Generation layer: free LLM providers, citation-strict prompts, answer composer."""

from __future__ import annotations

from .answerer import Answerer, compose_extractive
from .citations import parse_markers, sanitize_answer, sentences_with_markers, split_sentences
from .llm import GeminiLLM, GroqLLM, LLMResponse, OllamaLLM, describe_provider, get_llm
from .prompts import REFUSAL_TOKEN, SYSTEM_PROMPT, build_answer_prompt

__all__ = [
    "Answerer",
    "GeminiLLM",
    "GroqLLM",
    "LLMResponse",
    "OllamaLLM",
    "REFUSAL_TOKEN",
    "SYSTEM_PROMPT",
    "build_answer_prompt",
    "compose_extractive",
    "describe_provider",
    "get_llm",
    "parse_markers",
    "sanitize_answer",
    "sentences_with_markers",
    "split_sentences",
]
