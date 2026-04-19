"""Custom exception hierarchy for the ValueInvestor project.

All project-specific exceptions inherit from :class:`ValueInvestorError` so
callers can catch a single base class when broad error handling is desired.
"""

from __future__ import annotations


class ValueInvestorError(Exception):
    """Base exception for all ValueInvestor errors."""


class DataFetchError(ValueInvestorError):
    """Raised when fetching market/financial data fails critically."""


class ScreeningError(ValueInvestorError):
    """Raised when the screening pipeline encounters a fatal error."""


class AnalysisError(ValueInvestorError):
    """Raised when the analysis pipeline fails for a company."""


class LLMError(ValueInvestorError):
    """Raised on unrecoverable LLM / API errors."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider returns a rate-limit response after retries."""


class LLMAuthError(LLMError):
    """Raised when LLM authentication fails (invalid or missing API key)."""


class ReportGenerationError(ValueInvestorError):
    """Raised when report generation or rendering fails."""


class CacheError(ValueInvestorError):
    """Raised when the SQLite cache encounters a read/write error."""


class ConfigError(ValueInvestorError):
    """Raised when application configuration is invalid or missing."""
