from agents import WebSearchTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional, Dict, Any
from openai.types.shared.reasoning import Reasoning

# Tool definitions
web_search_preview = WebSearchTool(
    search_context_size="medium",
    user_location={"type": "approximate"}
)

# --- Schemas ---
class TriageSchema(BaseModel):
    has_all_details: bool = Field(description="Whether all required fields are present")
    initiative_goal: Optional[str] = Field(default=None, description="User-provided goal")
    target_timeframe: Optional[str] = Field(default=None, description="User-provided date or period")
    current_resources: Optional[str] = Field(default=None, description="User-provided resources")

# --- Agents ---
triage = Agent(
    name="Triage",
    instructions="""You are an assistant that gathers the key details needed to create a business initiative plan.

Look through the conversation to extract the following:
1. Initiative goal (what the team or organization aims to achieve)
2. Target completion date or timeframe
3. Available resources or current capacity (e.g., headcount, budget, or tool access)

If all three details are present anywhere in the conversation, return:
{
  "has_all_details": true,
  "initiative_goal": "<user-provided goal>",
  "target_timeframe": "<user-provided date or period>",
  "current_resources": "<user-provided resources>"
}
If one or more are missing, return:
{
  "has_all_details": false,
  "initiative_goal": "<goal if known or null>",
  "target_timeframe": "<timeframe if known or null>",
  "current_resources": "<resources if known or null>"
}""",
    model="gpt-5",
    output_type=TriageSchema,
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="minimal", summary="auto")
    )
)

launch_helper = Agent(
    name="Launch helper",
    instructions="""Come up with a tailored plan to help the user run a new business initiative.
Consider all the details they've provided and offer a succinct, bullet point list for how to run the initiative.

Use the web search tool to get additional context and synthesize a succinct answer that clearly explains how to run the project,
identifying unique opportunities, highlighting risks and laying out mitigations that make sense.""",
    model="gpt-4.1-mini",
    tools=[web_search_preview],
    model_settings=ModelSettings(
        temperature=1,
        top_p=1,
        max_tokens=2048,
        store=True
    )
)

get_data = Agent(
    name="Get data",
    instructions="""Collect the missing data from the user.

Look through the conversation to extract the following:
1. Initiative goal (what the team or organization aims to achieve)
2. Target completion date or timeframe
3. Available resources or current capacity (e.g., headcount, budget, or tool access)

Ask concise, direct questions to obtain whatever is missing.""",
    model="gpt-5",
    model_settings=ModelSettings(
        store=True,
        reasoning=Reasoning(effort="minimal", summary="auto")
    )
)

class WorkflowInput(BaseModel):
    input_as_text: str

# --- Main workflow ---
async def run_workflow(workflow_input: WorkflowInput) -> Dict[str, Any]:
    with trace("Agent builder workflow"):
        workflow = workflow_input.model_dump()
        conversation_history: list[TResponseInputItem] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": workflow["input_as_text"]}]
            }
        ]

        # 1) TRIAGE
        try:
            triage_result_temp = await Runner.run(
                triage,
                input=[*conversation_history],
                run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"})
            )
        except Exception as e:
            return {
                "ok": False,
                "stage": "triage",
                "error": f"Triage agent failed: {e}"
            }

        conversation_history.extend([item.to_input_item() for item in triage_result_temp.new_items])

        # Pydantic-safe parsing
        try:
            triage_output: TriageSchema = triage_result_temp.final_output
            triage_parsed = triage_output.model_dump()
        except ValidationError as ve:
            return {
                "ok": False,
                "stage": "triage-parse",
                "error": f"Triage output validation failed: {ve}"
            }

        result_blob: Dict[str, Any] = {
            "ok": True,
            "triage": {
                "output_text": triage_output.model_dump_json(),
                "output_parsed": triage_parsed
            }
        }

        # 2) BRANCH
        if triage_parsed.get("has_all_details", False):
            try:
                launch_helper_result_temp = await Runner.run(
                    launch_helper,
                    input=[*conversation_history],
                    run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"})
                )
                conversation_history.extend([item.to_input_item() for item in launch_helper_result_temp.new_items])
                result_blob.update({
                    "result_type": "launch_helper",
                    "result": {"output_text": launch_helper_result_temp.final_output_as(str)}
                })
            except Exception as e:
                result_blob.update({
                    "ok": False,
                    "stage": "launch_helper",
                    "error": f"Launch helper failed: {e}"
                })
        else:
            try:
                get_data_result_temp = await Runner.run(
                    get_data,
                    input=[*conversation_history],
                    run_config=RunConfig(trace_metadata={"__trace_source__": "agent-builder"})
                )
                conversation_history.extend([item.to_input_item() for item in get_data_result_temp.new_items])
                result_blob.update({
                    "result_type": "get_data",
                    "result": {"output_text": get_data_result_temp.final_output_as(str)}
                })
            except Exception as e:
                result_blob.update({
                    "ok": False,
                    "stage": "get_data",
                    "error": f"Get data failed: {e}"
                })

        # 3) RETURN everything useful
        result_blob["history"] = conversation_history
        return result_blob
