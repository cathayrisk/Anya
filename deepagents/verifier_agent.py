# Anya/deepagents/verifier_agent.py
from __future__ import annotations

from agents import Agent
from .models import VerificationResult

VERIFIER_PROMPT = """
你是驗證助理。給定某一步驟的輸出與 acceptance_criteria：
- 若 criteria.type 為 "research"，檢查是否至少有 min_sources 個來源、關鍵敘述是否一致
- 若 criteria.type 為 "code"，檢查提供的測試與預期結果是否完備
- 若 criteria.type 為 "data"，檢查輸出是否符合 schema
通過則 passed=true，否則 passed=false 並詳述 issues
輸出為 VerificationResult。
"""

verifier_agent = Agent(
    name="VerifierAgent",
    instructions=VERIFIER_PROMPT,
    model="gpt-4.1",
    output_type=VerificationResult,
)
