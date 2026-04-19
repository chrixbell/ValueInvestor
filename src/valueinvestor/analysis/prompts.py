"""LLM prompt templates for value investing analysis dimensions."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a senior value investing analyst specializing in Chinese equities "
    "(A-share and Hong Kong markets).\n\n"
    "Your philosophy: find good businesses that are misunderstood by the market — "
    "companies valued on false assumptions or wrong valuation models. "
    "You look for durable competitive advantages, capable management, and a clear "
    "gap between market perception and business reality.\n\n"
    "Requirements:\n"
    "- Provide clear, concise analysis backed by specific evidence from the data.\n"
    "- Always respond in valid JSON. No markdown, no commentary outside the JSON object.\n"
    "- When producing the JSON object, use Simplified Chinese (zh-CN) for all textual fields."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: object, suffix: str = "") -> str:
    """Format a numeric value for display, returning 'N/A' when missing."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def _pct(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return str(value)


def _build_company_block(data: dict) -> str:
    """Render a human-readable summary of the company data for the prompt."""
    company = data.get("company", {})
    financials = data.get("financials", {})
    valuation = data.get("valuation", {})

    lines = [
        f"Ticker: {company.get('ticker', 'N/A')}",
        f"Name: {company.get('name', 'N/A')}",
        f"Market: {company.get('market', 'N/A')}",
        f"Sector: {company.get('sector', 'N/A')}",
        f"Industry: {company.get('industry', 'N/A')}",
    ]

    desc = company.get("description")
    if desc:
        lines.append(f"Description: {desc}")

    lines.append("")
    lines.append("--- Financial Metrics ---")
    lines.append(f"Revenue: {_fmt(financials.get('revenue'))} RMB")
    lines.append(f"Net Income: {_fmt(financials.get('net_income'))} RMB")
    lines.append(f"Gross Margin: {_pct(financials.get('gross_margin'))}")
    lines.append(f"Net Margin: {_pct(financials.get('net_margin'))}")
    lines.append(f"ROE: {_pct(financials.get('roe'))}")
    lines.append(f"ROA: {_pct(financials.get('roa'))}")
    lines.append(f"Debt-to-Equity: {_fmt(financials.get('debt_to_equity'))}")
    lines.append(f"Free Cash Flow: {_fmt(financials.get('free_cash_flow'))} RMB")

    lines.append("")
    lines.append("--- Valuation Metrics ---")
    lines.append(f"Price: {_fmt(valuation.get('price'))}")
    lines.append(f"P/E (trailing): {_fmt(valuation.get('pe_ratio'))}")
    lines.append(f"P/E (forward): {_fmt(valuation.get('pe_forward'))}")
    lines.append(f"P/B: {_fmt(valuation.get('pb_ratio'))}")
    lines.append(f"P/S: {_fmt(valuation.get('ps_ratio'))}")
    lines.append(f"PEG: {_fmt(valuation.get('peg_ratio'))}")
    lines.append(f"Dividend Yield: {_pct(valuation.get('dividend_yield'))}")
    lines.append(f"EV/EBITDA: {_fmt(valuation.get('ev_to_ebitda'))}")
    lines.append(f"Market Cap: {_fmt(valuation.get('market_cap_rmb'))} RMB")

    scores = data.get("scores", {})
    if scores:
        lines.append("")
        lines.append("--- Screening Scores ---")
        lines.append(f"Composite: {_fmt(scores.get('composite_score'))}")
        lines.append(f"Value: {_fmt(scores.get('value_score'))}")
        lines.append(f"Quality: {_fmt(scores.get('quality_score'))}")
        lines.append(f"Growth: {_fmt(scores.get('growth_score'))}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dimension prompt functions
# ---------------------------------------------------------------------------

def prompt_business_nature(data: dict) -> str:
    """Build the user prompt for the business-nature dimension."""
    block = _build_company_block(data)
    return (
        "Analyze this company's business. Explain: What does the company do? "
        "What do they sell? Who is buying? "
        "Use simple words and a vivid metaphor if possible.\n\n"
        "Respond in JSON:\n"
        '{"title": "...", "content": "...", "confidence": 0.0-1.0}\n\n'
        f"Company data:\n{block}"
    )


def prompt_management(data: dict) -> str:
    """Build the user prompt for the management dimension."""
    block = _build_company_block(data)
    return (
        "Evaluate the company's leadership. Is management capable of: "
        "(a) leading the company to a position with very few competitors, "
        "(b) executing fast and at scale, "
        "(c) correcting mistakes and getting back on track? "
        "Provide one concrete instance for each point where evidence is available.\n\n"
        "Respond in JSON:\n"
        '{"title": "...", "content": "...", "confidence": 0.0-1.0}\n\n'
        f"Company data:\n{block}"
    )


def prompt_moat(data: dict) -> str:
    """Build the user prompt for the economic-moat dimension."""
    block = _build_company_block(data)
    return (
        "Analyze competitive advantages. Why do customers buy from this business? "
        "Who are the major competitors and what do they do? "
        "Does this company have real moats — and if so, what are they?\n\n"
        "Respond in JSON:\n"
        '{"title": "...", "content": "...", "confidence": 0.0-1.0}\n\n'
        f"Company data:\n{block}"
    )


def prompt_market_narrative(data: dict) -> str:
    """Build the user prompt for the market-narrative dimension."""
    block = _build_company_block(data)
    return (
        "Analyze the market narrative. What has the market believed about this "
        "company? Where are we in the timestamp of this story? What is the key "
        "insight that reality shows vs. market belief? What is most likely going "
        "to happen? What caused the key differences between perception and reality?\n\n"
        "Respond in JSON:\n"
        '{"title": "...", "content": "...", "confidence": 0.0-1.0}\n\n'
        f"Company data:\n{block}"
    )


def prompt_recommendation(data: dict) -> str:
    """Build the user prompt for the investment-recommendation dimension."""
    block = _build_company_block(data)
    return (
        "Based on all the analysis above, make an investment recommendation. "
        "Summarize the reasoning. Suggest portfolio weight (as a percentage of a "
        "total 500,000 RMB portfolio). List 3-5 key milestone events to watch for.\n\n"
        "Respond in JSON:\n"
        '{"title": "...", "content": "...", "confidence": 0.0-1.0, '
        '"portfolio_weight": 0.0-1.0, "milestones": ["..."]}\n\n'
        f"Company data:\n{block}"
    )


# ---------------------------------------------------------------------------
# Mapping of dimension name → prompt builder
# ---------------------------------------------------------------------------

DIMENSION_PROMPTS = {
    "business_nature": prompt_business_nature,
    "management": prompt_management,
    "moat": prompt_moat,
    "market_narrative": prompt_market_narrative,
    "recommendation": prompt_recommendation,
}
