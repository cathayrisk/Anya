# Anya/deepagents/models.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Step(BaseModel):
    id: str
    description: str
    is_parallel: bool = False
    parallel_group: Optional[str] = None
    requires_tool: bool = False
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = {}
    acceptance_criteria: Dict[str, Any] = {}
    expected_output_schema: Optional[Dict[str, Any]] = None
    max_retries: int = 1
    timeout: int = 60
    risk_level: str = "Low"
    notes: Optional[str] = None


class PlanMetadata(BaseModel):
    goal: str
    constraints: Optional[List[str]] = None
    estimated_cost: Optional[float] = None
    estimated_time_sec: Optional[int] = None
    budget: Optional[float] = None
    deadline: Optional[str] = None
    acceptance_criteria_final: Optional[Dict[str, Any]] = None
    sources: List[str] = []


class Plan(BaseModel):
    steps: List[Step]
    metadata: PlanMetadata


class TriageDecision(BaseModel):
    category: str             # "research" | "code" | "data" | "other"
    complexity: str           # "low" | "medium" | "high"
    approach: str             # "cot" | "subgoals" | "self_critique"
    recommended_tools: List[str] = []
    notes: Optional[str] = None


class VerificationResult(BaseModel):
    passed: bool
    issues: str = ""
