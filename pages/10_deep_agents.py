# Anya/pages/main.py
from __future__ import annotations

import os
import json
import asyncio
import random
from typing import Dict, List, Sequence, Tuple

import streamlit as st

st.set_page_config(page_title="Anya DeepAgents Orchestrator", page_icon="🧠")
st.title("🧠 Anya DeepAgents Orchestrator")
st.caption("A+ 版（小並行＋重試＋驗收）｜以 Streamlit 聊天互動執行 triage → plan → execute → verify → deliver")

# === 0.1 取得 API Key ===
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if not OPENAI_API_KEY:
    st.error("找不到 OpenAI API Key，請在 .streamlit/secrets.toml 設定 OPENAI_API_KEY 或 OPENAI_KEY。")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # 讓 Agents SDK 可以讀到

# 基礎套件
try:
    from agents import Agent, Runner
except Exception as e:
    st.error(f"無法載入基礎 agents 套件：{e}")
    st.stop()

# 自訂 Agents
try:
    from deepagents import (
        Step,
        Plan,
        TriageDecision,
        VerificationResult,
        triage_agent,
        planner_agent,
        research_agent,
        code_agent,
        data_agent,
        verifier_agent,
        writer_agent,
    )
except Exception as e:
    st.error("無法從 deepagents 載入自訂代理與型別，請確認路徑 Anya/deepagents 是否正確。錯誤：{}".format(e))
    st.stop()

# 工具：把 JSON 字串安全轉 dict
def _ensure_dict(obj) -> Dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {}
    return {}

# Orchestrator
class APlusOrchestrator:
    def __init__(self, max_parallel: int = 3, base_backoff: float = 1.0):
        self.max_parallel = max_parallel
        self.base_backoff = base_backoff

    async def run(self, goal: str) -> Dict[str, object]:
        # 1) Triage
        triage_res = await Runner.run(triage_agent, goal)
        triage = triage_res.final_output_as(TriageDecision)

        # 2) Plan
        planner_input = f"Goal: {goal}\nTriage: {triage.model_dump_json()}"
        plan_res = await Runner.run(planner_agent, planner_input)
        plan = plan_res.final_output_as(Plan)

        # 3) Execute
        outputs: Dict[str, str] = {}

        # 3a) 並行批次
        for _, steps in self._group_parallel_steps(plan.steps).items():
            await self._execute_parallel_batch(steps, outputs)

        # 3b) 序列步驟
        serial_steps = [s for s in plan.steps if not s.is_parallel]
        for step in serial_steps:
            sid, out = await self._execute_with_retry(step)
            outputs[sid] = out

        # 4) Writer
        writer_input = f"Goal: {plan.metadata.goal}\nArtifacts: {outputs}"
        final_res = await Runner.run(writer_agent, writer_input)
        final_output = str(final_res.final_output)

        # 5) 最終驗證
        final_criteria = _ensure_dict(plan.metadata.acceptance_criteria_final)
        verify_input = {"output": final_output, "criteria": final_criteria}
        final_ver = await Runner.run(verifier_agent, verify_input)
        verification = final_ver.final_output_as(VerificationResult)

        return {
            "ok": True,
            "triage": triage,
            "plan": plan,
            "step_outputs": outputs,
            "final_output": final_output,
            "verification": verification,
        }

    def _group_parallel_steps(self, steps: Sequence[Step]) -> Dict[str, List[Step]]:
        groups: Dict[str, List[Step]] = {}
        for s in steps:
            if s.is_parallel:
                key = s.parallel_group or "default_parallel"
                groups.setdefault(key, []).append(s)
        return groups

    async def _execute_parallel_batch(self, steps: Sequence[Step], outputs: Dict[str, str]) -> None:
        sem = asyncio.Semaphore(self.max_parallel)

        async def run_one(step: Step) -> Tuple[str, str]:
            async with sem:
                sid, out = await self._execute_with_retry(step)
                return sid, out

        tasks = [asyncio.create_task(run_one(s)) for s in steps]
        for coro in asyncio.as_completed(tasks):
            sid, out = await coro
            outputs[sid] = out

    async def _execute_with_retry(self, step: Step) -> Tuple[str, str]:
        attempts = step.max_retries + 1
        for i in range(attempts):
            try:
                output = await self._execute_step(step)
                # 驗收（把 JSON 字串轉 dict）
                criteria = _ensure_dict(step.acceptance_criteria)
                verify_input = {"output": output, "criteria": criteria}
                ver_res = await Runner.run(verifier_agent, verify_input)
                ver = ver_res.final_output_as(VerificationResult)
                if ver.passed:
                    return step.id, output
                else:
                    if i < attempts - 1:
                        await asyncio.sleep(self._backoff(i))
                        continue
                    raise RuntimeError(f"Step {step.id} failed verification: {ver.issues}")
            except Exception as e:
                if i < attempts - 1:
                    await asyncio.sleep(self._backoff(i))
                    continue
                raise RuntimeError(f"Step {step.id} error after retries: {e}")

    def _backoff(self, attempt: int) -> float:
        return (2 ** attempt) * self.base_backoff + random.uniform(0, 0.3)

    async def _execute_step(self, step: Step) -> str:
        agent = self._route_agent(step)
        input_payload = f"Step: {step.description}\nParams: {step.parameters}"
        res = await Runner.run(agent, input_payload)
        return str(res.final_output)

    def _route_agent(self, step: Step) -> Agent:
        if step.requires_tool and step.tool_name == "web_search":
            return research_agent
        if step.requires_tool and step.tool_name == "data_transform":
            return data_agent
        if step.requires_tool and step.tool_name == "code_run":
            return code_agent
        return research_agent

# Chat 狀態
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "嗨嗨～請描述你的目標或要解的問題，安妮亞幫你規劃→並行研究→彙整交付！🥜"}
    ]

# 側邊欄
with st.sidebar:
    st.header("設定")
    max_parallel = st.slider("最大並行數", min_value=1, max_value=8, value=3, step=1)
    base_backoff = st.slider("重試基礎退避秒數", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    st.caption("提示：並行數建議 2–4，退避越長越保守喔。")

# transcript
def transcript_from_messages(msgs: List[Dict[str, str]]) -> str:
    lines = []
    for m in msgs:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)

# 歷史訊息
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🤩" if m["role"] == "user" else "🧠"):
        st.markdown(m["content"])

# Chat input
prompt = st.chat_input("請輸入你的目標或要解的問題（可持續補充）", max_chars=2000, key="chat_input_main")

# 執行
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🤩"):
        st.markdown(prompt)

    full_text = transcript_from_messages(st.session_state.messages)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("安妮亞努力規劃與研究中…(滴答滴答)"):
            async def _run_once() -> Dict[str, object]:
                orchestrator = APlusOrchestrator(max_parallel=max_parallel, base_backoff=base_backoff)
                return await orchestrator.run(full_text)

            try:
                out = asyncio.run(_run_once())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    out = loop.run_until_complete(_run_once())
                finally:
                    loop.close()
            except Exception as e:
                st.error(f"流程失敗：{e}")
                st.stop()

        # Triage
        tri = out.get("triage")
        if tri:
            with st.expander("Triage 摘要", expanded=False):
                st.markdown(f"- category: {tri.category}")
                st.markdown(f"- complexity: {tri.complexity}")
                st.markdown(f"- approach: {tri.approach}")
                if tri.recommended_tools:
                    st.markdown(f"- recommended_tools: {', '.join(tri.recommended_tools)}")
                if tri.notes:
                    st.markdown(f"- notes: {tri.notes}")

        # Plan
        plan = out.get("plan")
        if plan:
            with st.expander("Plan 步驟（含並行標註）", expanded=False):
                for i, s in enumerate(plan.steps, start=1):
                    tag = "並行" if s.is_parallel else "序列"
                    tool = f"{s.tool_name}" if s.tool_name else "-"
                    st.markdown(f"**Step {i} | {s.id}** · {tag} · tool={tool}")
                    st.markdown(f"- {s.description}")
                    if s.parallel_group:
                        st.markdown(f"- parallel_group: {s.parallel_group}")
                    if s.acceptance_criteria is not None:
                        show = s.acceptance_criteria if isinstance(s.acceptance_criteria, str) else json.dumps(s.acceptance_criteria, ensure_ascii=False)
                        st.markdown(f"- acceptance_criteria: `{show}`")
                    if s.max_retries or s.timeout:
                        st.caption(f"retries={s.max_retries}, timeout={s.timeout}s")

        # 步驟輸出
        step_outputs = out.get("step_outputs") or {}
        if step_outputs:
            with st.expander("步驟輸出（摘要）", expanded=False):
                for sid, text in step_outputs.items():
                    st.markdown(f"**{sid}**")
                    st.code(text[:2000] + ("..." if len(text) > 2000 else ""), language="markdown")

        # 最終輸出
        final_output = out.get("final_output") or ""
        st.markdown("### 最終結果")
        st.write(final_output)

        # 最終驗證
        ver = out.get("verification")
        if ver:
            ok_emoji = "✅" if ver.passed else "⚠️"
            with st.expander(f"最終驗證 {ok_emoji}", expanded=not ver.passed):
                st.markdown(f"- passed: {ver.passed}")
                if ver.issues:
                    st.markdown(f"- issues: {ver.issues}")

        st.session_state.messages.append({"role": "assistant", "content": final_output or "(流程完成)（無最終輸出）"})
