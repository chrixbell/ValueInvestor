"""OpenAI API client wrapper for ValueInvestor LLM-driven analysis."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

import openai
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

from valueinvestor.config import LLMConfig
from valueinvestor.data.models import AnalysisDimension
from valueinvestor.errors import LLMAuthError, LLMError, LLMRateLimitError

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD) – updated for known models.
_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
    # OpenAI models via GitHub
    "openai/gpt-4.1": {"input": 2.00, "output": 8.00},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Anthropic Claude via GitHub Copilot
    "claude-sonnet-4.6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-opus-4.6": {"input": 15.00, "output": 75.00},
    "anthropic/claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    # Legacy Copilot model naming
    "Copilot/Claude Sonnet 4.6": {"input": 3.00, "output": 15.00},
    "Copilot/Claude Opus 4.6": {"input": 15.00, "output": 75.00},
    "Copilot/ChatGPT 5.4": {"input": 5.00, "output": 15.00},
    # Gemini models
    "gemini-pro-3.1": {"input": 1.25, "output": 3.75},
    "gemini-1.5-pro": {"input": 1.25, "output": 3.75},
    "gemini-1.5-pro-latest": {"input": 1.25, "output": 3.75},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-3.1-flash-lite-preview": {"input": 0.075, "output": 0.30},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 3.75},
    # Kimi
    "Kimi Code 2.5": {"input": 1.00, "output": 2.00},
    # NVIDIA NIM (Minimax M2.7)
    "minimax-m2-7": {"input": 0.30, "output": 0.60},
    # Local LLM (free, no cost)
    "local_llm": {"input": 0.00, "output": 0.00},
}


_DIMENSION_SYSTEM_PROMPTS: Dict[str, str] = {
    "business_nature": (
        "You are a senior equity research analyst specialising in Chinese stock markets. "
        "Analyse the provided company data and describe the nature of its business. "
        "Cover its core products/services, revenue drivers, industry position, and "
        "competitive landscape. Be concise yet thorough."
    ),
    "management": (
        "You are a senior equity research analyst specialising in Chinese stock markets. "
        "Evaluate the company's management quality based on the provided data. "
        "Consider capital allocation track record, insider ownership, governance, "
        "and alignment with minority shareholders."
    ),
    "moat": (
        "You are a senior equity research analyst specialising in Chinese stock markets. "
        "Assess the company's economic moat. Consider brand strength, switching costs, "
        "network effects, cost advantages, intangible assets, and efficient scale."
    ),
    "market_narrative": (
        "You are a senior equity research analyst specialising in Chinese stock markets. "
        "Analyse the current market narrative and sentiment around this company. "
        "Consider recent news, analyst consensus, sector trends, and potential catalysts "
        "or risks."
    ),
    "recommendation": (
        "You are a senior equity research analyst specialising in Chinese stock markets. "
        "Based on the provided company data, give a clear investment recommendation. "
        "Include a target price range, expected return, key risks, and suggested "
        "position sizing rationale."
    ),
}

_DIMENSION_TITLES: Dict[str, str] = {
    "business_nature": "Business Nature & Industry Analysis",
    "management": "Management Quality Assessment",
    "moat": "Economic Moat Evaluation",
    "market_narrative": "Market Narrative & Sentiment",
    "recommendation": "Investment Recommendation",
}


class LLMClient:
    """Wrapper around the OpenAI Python SDK for ValueInvestor analysis tasks."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        temperature: float = 0.3,
        config: Optional[LLMConfig] = None,
    ) -> None:
        base_url = None
        self.provider = "openai"
        if config is not None:
            api_key = config.api_key or api_key
            model = config.model
            max_retries = config.max_retries
            temperature = config.temperature
            base_url = config.base_url
            self.provider = config.provider

        if not api_key:
            provider_name = config.provider if config else "openai"
            raise LLMAuthError(
                f"No API key provided for {provider_name}. Check your environment variables or config."
            )

        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature

        if self.provider == "gemini":
            self._gemini_client = genai.Client(api_key=api_key)
        elif self.provider == "local_llm":
            # For local LLM, base_url must include /v1 prefix for OpenAI-compatible API
            base = (base_url or "http://127.0.0.1:1234").rstrip("/")
            if not base.endswith("/v1"):
                base = base + "/v1"
            self._base_url = base
            # Create OpenAI client pointing to local LLM server (OpenAI-compatible API)
            self._client = OpenAI(api_key="not-needed", base_url=self._base_url, max_retries=0)
        else:
            # The SDK's own retry is disabled so we handle backoff ourselves.
            self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

        # Cumulative usage tracking
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate USD cost for a given token count based on model pricing."""
        pricing = _MODEL_PRICING.get(self.model, _MODEL_PRICING["gpt-4o"])
        input_cost = prompt_tokens * pricing["input"] / 1_000_000
        output_cost = completion_tokens * pricing["output"] / 1_000_000
        return input_cost + output_cost

    def get_usage_summary(self) -> dict:
        """Return cumulative usage statistics."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "model": self.model,
            "provider": self.provider,
        }

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    def _complete_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None,
    ) -> str:
        generation_config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json" if response_format == "json" else None,
        )

        # Combine system prompt and user prompt
        prompt = f"{system_prompt}\n\n{user_prompt}"

        last_exception: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=generation_config,
                )
                
                # Track usage
                usage = response.usage_metadata
                if usage:
                    prompt_tok = usage.prompt_token_count
                    completion_tok = usage.candidates_token_count
                    cost = self.estimate_cost(prompt_tok, completion_tok)

                    self.total_prompt_tokens += prompt_tok
                    self.total_completion_tokens += completion_tok
                    self.total_cost_usd += cost

                    logger.info(
                        "Token usage – prompt: %d, completion: %d, cost: $%.4f",
                        prompt_tok,
                        completion_tok,
                        cost,
                    )

                return response.text
            except Exception as exc:
                last_exception = exc
                wait = 2 ** attempt
                logger.warning(
                    "Gemini API error (attempt %d/%d): %s. Retrying in %ds …",
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        else:
            logger.error("Gemini request failed after %d attempts.", self.max_retries)
            raise LLMError(
                f"Gemini request failed after {self.max_retries} attempts: {last_exception}"
            ) from last_exception


    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None,
    ) -> str:
        """Send a chat completion request with retry and usage tracking.

        Parameters
        ----------
        system_prompt:
            The system-level instruction.
        user_prompt:
            The user message / data payload.
        response_format:
            If ``"json"``, requests a JSON-formatted response from the API.

        Returns
        -------
        str
            The assistant's response content.
        """
        if self.provider == "gemini":
            return self._complete_gemini(system_prompt, user_prompt, response_format)

        # Handle local_llm and OpenAI-compatible APIs (OpenAI, GitHub, OpenRouter, NVIDIA NIM, etc.)
        messages: list[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        last_exception: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except openai.AuthenticationError as exc:
                logger.error(
                    "OpenAI authentication failed. Verify your API key is correct."
                )
                raise LLMAuthError(str(exc)) from exc
            except openai.RateLimitError as exc:
                last_exception = exc
                wait = 2 ** attempt
                logger.warning(
                    "Rate-limited by OpenAI (attempt %d/%d). Retrying in %ds …",
                    attempt,
                    self.max_retries,
                    wait,
                )
                time.sleep(wait)
            except openai.APIError as exc:
                last_exception = exc
                wait = 2 ** attempt
                logger.warning(
                    "OpenAI API error (attempt %d/%d): %s. Retrying in %ds …",
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        else:
            # All retries exhausted
            logger.error(
                "OpenAI request failed after %d attempts.", self.max_retries
            )
            if isinstance(last_exception, openai.RateLimitError):
                raise LLMRateLimitError(
                    f"Rate-limited after {self.max_retries} retries"
                ) from last_exception
            raise LLMError(
                f"OpenAI request failed after {self.max_retries} attempts: {last_exception}"
            ) from last_exception

        # Track usage
        usage = response.usage
        if usage:
            prompt_tok = usage.prompt_tokens
            completion_tok = usage.completion_tokens
            cost = self.estimate_cost(prompt_tok, completion_tok)

            self.total_prompt_tokens += prompt_tok
            self.total_completion_tokens += completion_tok
            self.total_cost_usd += cost

            logger.info(
                "Token usage – prompt: %d, completion: %d, cost: $%.4f",
                prompt_tok,
                completion_tok,
                cost,
            )

        # Defensive: check response structure before subscripting
        if not response or not response.choices or len(response.choices) == 0:
            logger.error("Malformed LLM response: choices is empty or missing. Full response: %s", response)
            raise LLMError(f"Malformed LLM response: no choices returned")
        
        if not response.choices[0].message:
            logger.error("Malformed LLM response: message is missing from choice 0. Full response: %s", response)
            raise LLMError(f"Malformed LLM response: no message in choice")
        
        content = response.choices[0].message.content or ""
        return content

    # ------------------------------------------------------------------
    # High-level analysis
    # ------------------------------------------------------------------

    def analyze_company(
        self, company_data: dict, dimension: str
    ) -> AnalysisDimension:
        """Analyse a company along one dimension and return a structured result.

        Parameters
        ----------
        company_data:
            Dictionary of company/financial information to feed the LLM.
        dimension:
            One of ``"business_nature"``, ``"management"``, ``"moat"``,
            ``"market_narrative"``, ``"recommendation"``.
        """
        if dimension not in _DIMENSION_SYSTEM_PROMPTS:
            raise ValueError(
                f"Unknown dimension {dimension!r}. "
                f"Must be one of {list(_DIMENSION_SYSTEM_PROMPTS)}"
            )

        system_prompt = _DIMENSION_SYSTEM_PROMPTS[dimension]
        user_prompt = (
            "Respond in JSON with keys: \"title\", \"content\", \"confidence\".\n"
            "\"confidence\" is a float between 0.0 and 1.0 indicating your "
            "certainty in the analysis.\n\n"
            f"Company data:\n{json.dumps(company_data, ensure_ascii=False, indent=2)}"
        )

        raw = self.complete(system_prompt, user_prompt, response_format="json")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON: %s", raw[:200])
            # Fallback: wrap raw text as the content
            return AnalysisDimension(
                dimension=dimension,
                title=_DIMENSION_TITLES.get(dimension, dimension),
                content=raw,
            )

        return AnalysisDimension(
            dimension=dimension,
            title=parsed.get("title", _DIMENSION_TITLES.get(dimension, dimension)),
            content=parsed.get("content", raw),
            confidence=parsed.get("confidence"),
        )
