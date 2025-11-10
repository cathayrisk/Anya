# Anya/pages/main.py
from __future__ import annotations

# 先處理環境變數與靜音追蹤匯出（一定要在 import agents 之前）
import os
os.environ.setdefault("AGENTS_TRACE_EXPORT", "disabled")  # 關掉 trace export 初始化訊息

import json
import asyncio
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import streamlit as st

st.set_page_config(page_title="Anya DeepAgents Orchestrator", page_icon="🧠")
st.title("🧠 Anya DeepAgents Orchestrator")
st.caption("A+ 版（小並行＋重試＋驗收）｜以 Streamlit 聊天互動執行 triage → plan → execute → verify → deliver")

# === 取得 API Key（先環境後 secrets，並在 import agents 前完成設定）===
_openai_key = os.getenv("OPENAI_API_KEY")
_openai_key = st.secrets.get("OPENAI_API_KEY") or st.secrets["OPENAI_KEY"] or _openai_key
if not _openai_key:
    st.error("找不到 OpenAI API Key，請在 .streamlit/secrets.toml 設定 OPENAI_API_KEY 或 OPENAI_KEY。")
    st.stop()
os.environ["OPENAI_API_KEY"] = _openai_key  # 讓 Agents SDK 可在 import 後直接讀到

# 基礎套件（現在再載入，會讀到 OPENAI_API_KEY；且 trace 已被關掉）
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

# 預設最佳參數（無側欄）
DEFAULT_MAX_PARALLEL = 5          # 建議 4–6；取 5
DEFAULT_BASE_BACKOFF = 0.6        # 退避較靈敏
DEFAULT_STRICT_VERIFY = False     # 步驟驗收未過→以 [WARN] 放行，不中斷

# Orchestrator
class APlusOrchestrator:
    def __init__(
        self,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        base_backoff: float = DEFAULT_BASE_BACKOFF,
        strict_verify: bool = DEFAULT_STRICT_VERIFY,
        progress: Optional[Callable[[str, Dict], None]] = None,
    ):
        self.max_parallel = max_parallel
        self.base_backoff = base_backoff
        self.strict_verify = strict_verify
        self._progress = progress or (lambda *args, **kwargs: None)

    def _notify(self, event: str, **payload):
        try:
            self._progress(event, payload)
        except Exception:
            # 靜默忽略 UI 回報失敗，避免影響主流程
            pass

    async def run(self, goal: str) -> Dict[str, object]:
        # 1) Triage
        self._notify("triage.start", goal=goal)
        triage_res = await Runner.run(triage_agent, goal)
        triage = triage_res.final_output_as(TriageDecision)
        self._notify("triage.done", triage=triage)

        # 2) Plan
        self._notify("plan.start")
        planner_input = f"Goal: {goal}\nTriage: {triage.model_dump_json()}"
        plan_res = await Runner.run(planner_agent, planner_input)
        plan = plan_res.final_output_as(Plan)
        self._notify("plan.done", total_steps=len(plan.steps))

        # 3) Execute
        self._notify("execute.start")
        outputs: Dict[str, str] = {}

        # 3a) 並行批次
        for group_key, steps in self._group_parallel_steps(plan.steps).items():
            self._notify("execute.batch_start", batch=group_key, count=len(steps))
            await self._execute_parallel_batch(steps, outputs)
            self._notify("execute.batch_done", batch=group_key)

        # 3b) 序列步驟
        serial_steps = [s for s in plan.steps if not s.is_parallel]
        for step in serial_steps:
            self._notify("execute.step_start", step_id=step.id, desc=step.description, tool=step.tool_name)
            sid, out = await self._execute_with_retry(step)
            outputs[sid] = out
            # 根據是否 WARN/ERROR 更新
            if out and "[ERROR]" in out:
                self._notify("execute.step_error", step_id=sid, message=out)
            elif out and "[WARN]" in out:
                self._notify("execute.step_warn", step_id=sid, message=out)
            else:
                self._notify("execute.step_ok", step_id=sid)
        self._notify("execute.done")

        # 4) Writer
        self._notify("write.start")
        writer_input = f"Goal: {plan.metadata.goal}\nArtifacts: {outputs}"
        final_res = await Runner.run(writer_agent, writer_input)
        final_output = str(final_res.final_output)
        self._notify("write.done")

        # 5) 最終驗證（提供預設標準，避免缺 criteria）
        self._notify("final_verify.start")
        final_criteria = _ensure_dict(getattr(plan.metadata, "acceptance_criteria_final", None))
        if not final_criteria:
            final_criteria = {
                "type": "research",
                "min_sources": 8,
                "must_have_sections": ["政策公告彙整", "主流媒體交叉", "學者/產業觀點", "社會影響", "事件時間線"],
                "per_source_fields": ["title", "url", "published_date"],
                "date_window_max_months": 18,
            }
        verify_input = {"output": final_output, "criteria": final_criteria}
        final_ver = await Runner.run(verifier_agent, json.dumps(verify_input, ensure_ascii=False))
        verification = final_ver.final_output_as(VerificationResult)
        self._notify("final_verify.done", passed=verification.passed)

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
                try:
                    self._notify("execute.step_start", step_id=step.id, desc=step.description, tool=step.tool_name)
                    sid, out = await self._execute_with_retry(step)
                    # 回報狀態
                    if out and "[ERROR]" in out:
                        self._notify("execute.step_error", step_id=sid, message=out)
                    elif out and "[WARN]" in out:
                        self._notify("execute.step_warn", step_id=sid, message=out)
                    else:
                        self._notify("execute.step_ok", step_id=sid)
                    return sid, out
                except Exception as e:
                    self._notify("execute.step_error", step_id=step.id, message=str(e))
                    return step.id, f"[ERROR] {e}"

        tasks = [asyncio.create_task(run_one(s)) for s in steps]
        for coro in asyncio.as_completed(tasks):
            try:
                sid, out = await coro
            except Exception as e:
                sid, out = "unknown_step", f"[ERROR] {e}"
            outputs[sid] = out

    async def _execute_with_retry(self, step: Step) -> Tuple[str, str]:
        attempts = step.max_retries + 1 if getattr(step, "max_retries", None) is not None else 2  # 預設重試 1 次
        for i in range(attempts):
            try:
                output = await self._execute_step(step, attempt=i, total_attempts=attempts)
                # 驗收（把 JSON 字串轉 dict）
                criteria = _ensure_dict(step.acceptance_criteria)
                verify_input = {"output": output, "criteria": criteria}
                ver_res = await Runner.run(verifier_agent, json.dumps(verify_input, ensure_ascii=False))
                ver = ver_res.final_output_as(VerificationResult)
                if ver.passed:
                    return step.id, output
                else:
                    if i < attempts - 1:
                        await asyncio.sleep(self._backoff(i))
                        continue
                    if not self.strict_verify:
                        return step.id, f"{output}\n\n[WARN] verify failed: {ver.issues}"
                    raise RuntimeError(f"Step {step.id} failed verification: {ver.issues}")
            except Exception as e:
                if i < attempts - 1:
                    await asyncio.sleep(self._backoff(i))
                    continue
                return step.id, f"[ERROR] Step {step.id} error after retries: {e}"

    def _backoff(self, attempt: int) -> float:
        return (2 ** attempt) * self.base_backoff + random.uniform(0, 0.3)

    def _cap_timeout(self, step: Step) -> float:
        # 依步驟類型給合理上限，避免超長等待
        typ = _ensure_dict(step.acceptance_criteria).get("type", "")
        if typ == "research":
            cap = 90.0
        elif typ in ("data", "code"):
            cap = 120.0
        else:
            cap = 240.0  # 例如寫作或未標註型別
        t = getattr(step, "timeout", None)
        if t is None or t <= 0:
            return cap
        return min(float(t), cap)

    async def _execute_step(self, step: Step, attempt: int = 0, total_attempts: int = 1) -> str:
        agent = self._route_agent(step)
        criteria = _ensure_dict(step.acceptance_criteria)
        input_payload = (
            f"Step: {step.description}\n"
            f"Params: {step.parameters}\n"
            f"AcceptanceCriteria: {json.dumps(criteria, ensure_ascii=False)}\n"
            f"Retry: {attempt + 1}/{total_attempts}"
        )
        task = Runner.run(agent, input_payload)
        timeout = self._cap_timeout(step)
        res = await asyncio.wait_for(task, timeout=timeout)
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
        # 狀態列（st.status）即時顯示進度
        with st.status("分類中（Triage）", state="running") as tri_stat, \
             st.status("規劃中（Plan）", state="waiting") as plan_stat, \
             st.status("執行中（Execute）", state="waiting") as exec_stat, \
             st.status("撰寫中（Write）", state="waiting") as write_stat, \
             st.status("驗證中（Final Verify）", state="waiting") as final_stat:

            def _progress_cb(event: str, info: Dict):
                if event == "triage.start":
                    tri_stat.update(label="分類中（Triage）", state="running")
                elif event == "triage.done":
                    tri_stat.update(label="分類完成（Triage）", state="complete")

                elif event == "plan.start":
                    plan_stat.update(label="規劃中（Plan）", state="running")
                elif event == "plan.done":
                    plan_stat.update(label=f"規劃完成（{info.get('total_steps', 0)} 步）", state="complete")
                    exec_stat.update(label="執行中（Execute）", state="running")

                elif event == "execute.start":
                    exec_stat.update(label="執行中（Execute）", state="running")
                elif event == "execute.batch_start":
                    exec_stat.update(label=f"執行中：並行批次「{info.get('batch')}」", state="running")
                elif event == "execute.batch_done":
                    exec_stat.update(label=f"執行中：批次「{info.get('batch')}」完成", state="running")
                elif event == "execute.step_start":
                    # 可視需要寫更細緻訊息（略）
                    pass
                elif event == "execute.step_ok":
                    pass
                elif event == "execute.step_warn":
                    pass
                elif event == "execute.step_error":
                    pass
                elif event == "execute.done":
                    exec_stat.update(label="執行完成（Execute）", state="complete")
                    write_stat.update(label="撰寫中（Write）", state="running")

                elif event == "write.start":
                    write_stat.update(label="撰寫中（Write）", state="running")
                elif event == "write.done":
                    write_stat.update(label="撰寫完成（Write）", state="complete")
                    final_stat.update(label="驗證中（Final Verify）", state="running")

                elif event == "final_verify.start":
                    final_stat.update(label="驗證中（Final Verify）", state="running")
                elif event == "final_verify.done":
                    if info.get("passed"):
                        final_stat.update(label="驗證完成（✅ 通過）", state="complete")
                    else:
                        final_stat.update(label="驗證完成（⚠️ 有問題）", state="complete")

            async def _run_once() -> Dict[str, object]:
                orchestrator = APlusOrchestrator(
                    max_parallel=DEFAULT_MAX_PARALLEL,
                    base_backoff=DEFAULT_BASE_BACKOFF,
                    strict_verify=DEFAULT_STRICT_VERIFY,
                    progress=_progress_cb,
                )
                return await orchestrator.run(full_text)

            try:
                out = asyncio.run(_run_once())
            except RuntimeError:
                # Fallback: 在已存在 event loop 的環境
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    out = loop.run_until_complete(_run_once())
                except Exception as e:
                    final_stat.update(label=f"流程失敗：{e}", state="error")
                    st.error(f"流程失敗：{e}")
                    st.stop()
                finally:
                    loop.close()
            except Exception as e:
                final_stat.update(label=f"流程失敗：{e}", state="error")
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

        # Plan（不顯示 S1/S2…，只顯示「步驟編號」與描述）
        plan = out.get("plan")
        if plan:
            with st.expander("Plan 步驟（含並行標註）", expanded=False):
                for i, s in enumerate(plan.steps, start=1):
                    tag = "並行" if s.is_parallel else "序列"
                    tool = f"{s.tool_name}" if s.tool_name else "-"
                    st.markdown(f"**步驟 {i}** · {tag} · tool={tool}")
                    st.markdown(f"- {s.description}")
                    if s.parallel_group:
                        st.markdown(f"- parallel_group: {s.parallel_group}")
                    # 若需要可顯示驗收條件
                    # show = s.acceptance_criteria if isinstance(s.acceptance_criteria, str) else json.dumps(s.acceptance_criteria, ensure_ascii=False)
                    # st.markdown(f"- acceptance_criteria: `{show}`")
                    if getattr(s, "max_retries", None) is not None or getattr(s, 'timeout', None):
                        st.caption(f"retries={getattr(s, 'max_retries', 0)}, timeout={getattr(s, 'timeout', None)}s")

        # 步驟輸出（摘要）
        step_outputs = out.get("step_outputs") or {}
        if step_outputs:
            with st.expander("步驟輸出（摘要）", expanded=False):
                for sid, text in step_outputs.items():
                    # 不再顯示 S-id；僅顯示內容
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
