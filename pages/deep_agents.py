# Anya/pages/main.py
from __future__ import annotations

import os
os.environ.setdefault("AGENTS_TRACE_EXPORT", "disabled")

import json
import re
import time
import asyncio
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Set

import streamlit as st

st.set_page_config(page_title="Anya DeepAgents Orchestrator", page_icon="🧠")
st.title("🧠 Anya DeepAgents Orchestrator")
st.caption("A+ 版｜單狀態列 + 過程紀錄 + 假連結防呆 + 自動修訂直到通過（最多2回）")

# === API Key ===
_openai_key = os.getenv("OPENAI_API_KEY")
_openai_key = st.secrets.get("OPENAI_API_KEY") or st.secrets["OPENAI_KEY"] or _openai_key
if not _openai_key:
    st.error("找不到 OpenAI API Key，請在 .streamlit/secrets.toml 設定 OPENAI_API_KEY 或 OPENAI_KEY。")
    st.stop()
os.environ["OPENAI_API_KEY"] = _openai_key

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

# 小工具
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

_URL_RE = re.compile(r"https?://[^\s)>\]}]+", re.IGNORECASE)

def _extract_urls(text: str) -> List[str]:
    return _URL_RE.findall(text or "")

def _looks_like_fake_url(u: str) -> bool:
    bad = ("example.com", "localhost", "127.0.0.1")
    return any(b in u.lower() for b in bad)

def _has_fake_url(text: str) -> bool:
    return any(_looks_like_fake_url(u) for u in _extract_urls(text))

# 預設最佳參數
DEFAULT_MAX_PARALLEL = 5
DEFAULT_BASE_BACKOFF = 0.6
DEFAULT_STRICT_VERIFY = False  # 步驟未過先警告放行；最終階段有自動修訂循環

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
        self._progress = progress or (lambda *a, **k: None)
        self._step_order: Dict[str, int] = {}
        self._agent_usage: List[Dict[str, object]] = []
        self._agents_used: Set[str] = set()

    def _notify(self, event: str, **payload):
        try:
            self._progress(event, payload)
        except Exception:
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
        # 步驟編號表
        self._step_order = {s.id: i + 1 for i, s in enumerate(plan.steps)}
        self._notify("plan.done", total_steps=len(plan.steps))

        # 3) Execute
        self._notify("execute.start")
        outputs: Dict[str, str] = {}

        # 並行批次
        for group_key, steps in self._group_parallel_steps(plan.steps).items():
            self._notify("execute.batch_start", batch=group_key, count=len(steps))
            await self._execute_parallel_batch(steps, outputs)
            self._notify("execute.batch_done", batch=group_key)

        # 序列步驟
        serial_steps = [s for s in plan.steps if not s.is_parallel]
        for step in serial_steps:
            self._notify("execute.step_start", step_id=step.id, step_num=self._step_order.get(step.id), desc=step.description, tool=step.tool_name)
            sid, out = await self._execute_with_retry(step)
            outputs[sid] = out
            evt = "execute.step_ok"
            if out and "[ERROR]" in out:
                evt = "execute.step_error"
            elif out and "[WARN]" in out:
                evt = "execute.step_warn"
            self._notify(evt, step_id=sid, step_num=self._step_order.get(sid), desc=step.description, output=out)
        self._notify("execute.done")

        # 4) Writer
        self._notify("write.start")
        writer_input = f"Goal: {plan.metadata.goal}\nArtifacts: {outputs}"
        final_res = await Runner.run(writer_agent, writer_input)
        final_output = str(final_res.final_output)
        self._notify("write.done", output=final_output)

        # 5) Final verify + 修訂循環（最多 2 回）
        self._notify("final_verify.start")
        final_criteria = _ensure_dict(getattr(plan.metadata, "acceptance_criteria_final", None))
        if not final_criteria:
            final_criteria = {
                "type": "research",
                "min_sources": 8,
                "must_have_sections": ["政策公告彙整", "主流媒體交叉", "學者/產業觀點", "社會影響", "事件時間線"],
                "per_source_fields": ["title", "url", "published_date"],
                "date_window_max_months": 18,
                "forbid_domains": ["example.com", "localhost", "127.0.0.1"],
            }

        verification = await self._verify(final_output, final_criteria)
        rounds = 0
        while not verification.passed and rounds < 2:
            rounds += 1
            self._notify("revise.start", round=rounds, issues=verification.issues)
            # 把問題回饋給 writer 要求修正：補區塊、改連結、補日期等
            repair_prompt = (
                "請針對以下驗證意見修正輸出：\n"
                f"{verification.issues}\n\n"
                "重點要求：\n"
                "- 若有假連結或 example.com/localhost/127.0.0.1，請以真實、可點擊的原始來源替換（官方或原刊）。\n"
                "- 每則來源都要有 title/url/published_date（YYYY-MM-DD）並與文本對應。\n"
                "- 需包含且清楚標示以下段落標題：政策公告彙整｜主流媒體交叉｜學者/產業觀點｜社會影響｜事件時間線。\n"
                "- 若來源不足請主動補足至標準，並避免重複同一網址或同一網域首頁。\n"
                "以下為前一版輸出，請直接回傳修正後全文：\n"
            )
            repair_res = await Runner.run(writer_agent, repair_prompt + final_output)
            final_output = str(repair_res.final_output)
            verification = await self._verify(final_output, final_criteria)
            self._notify("revise.done", round=rounds, passed=verification.passed)
        self._notify("final_verify.done", passed=verification.passed, issues=verification.issues)

        return {
            "ok": True,
            "triage": triage,
            "plan": plan,
            "step_outputs": outputs,
            "final_output": final_output,
            "verification": verification,
            "agents_used": sorted(self._agents_used),
            "agent_usage": self._agent_usage,
        }

    async def _verify(self, output: str, criteria: Dict) -> VerificationResult:
        v_in = {"output": output, "criteria": criteria}
        ver = await Runner.run(verifier_agent, json.dumps(v_in, ensure_ascii=False))
        return ver.final_output_as(VerificationResult)

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
                    self._notify("execute.step_start", step_id=step.id, step_num=self._step_order.get(step.id), desc=step.description, tool=step.tool_name)
                    sid, out = await self._execute_with_retry(step)
                    evt = "execute.step_ok"
                    if out and "[ERROR]" in out:
                        evt = "execute.step_error"
                    elif out and "[WARN]" in out:
                        evt = "execute.step_warn"
                    self._notify(evt, step_id=sid, step_num=self._step_order.get(sid), desc=step.description, output=out)
                    return sid, out
                except Exception as e:
                    self._notify("execute.step_error", step_id=step.id, step_num=self._step_order.get(step.id), desc=step.description, output=str(e))
                    return step.id, f"[ERROR] {e}"

        tasks = [asyncio.create_task(run_one(s)) for s in steps]
        for coro in asyncio.as_completed(tasks):
            try:
                sid, out = await coro
            except Exception as e:
                sid, out = "unknown_step", f"[ERROR] {e}"
            outputs[sid] = out

    async def _execute_with_retry(self, step: Step) -> Tuple[str, str]:
        attempts = step.max_retries + 1 if getattr(step, "max_retries", None) is not None else 2
        for i in range(attempts):
            try:
                output = await self._execute_step(step, attempt=i, total_attempts=attempts)
                # 假網址防呆（在正式驗收前先擋掉）
                if _has_fake_url(output):
                    if i < attempts - 1:
                        await asyncio.sleep(self._backoff(i))
                        continue
                    if not self.strict_verify:
                        return step.id, f"{output}\n\n[WARN] 偵測到疑似假連結或占位連結，請更換為真實來源。"
                    raise RuntimeError("輸出含疑似假連結（example.com/localhost/127.0.0.1）")

                # 正式驗收
                criteria = self._merged_criteria(step)
                v_in = {"output": output, "criteria": criteria}
                ver_res = await Runner.run(verifier_agent, json.dumps(v_in, ensure_ascii=False))
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
        typ = _ensure_dict(step.acceptance_criteria).get("type", "")
        if typ == "research":
            cap = 90.0
        elif typ in ("data", "code"):
            cap = 120.0
        else:
            cap = 240.0
        t = getattr(step, "timeout", None)
        if t is None or t <= 0:
            return cap
        return min(float(t), cap)

    def _agent_label(self, step: Step) -> str:
        if step.requires_tool and step.tool_name == "web_search":
            return "research_agent"
        if step.requires_tool and step.tool_name == "data_transform":
            return "data_agent"
        if step.requires_tool and step.tool_name == "code_run":
            return "code_agent"
        return "research_agent"

    async def _execute_step(self, step: Step, attempt: int = 0, total_attempts: int = 1) -> str:
        agent = self._route_agent(step)
        agent_label = self._agent_label(step)
        self._agents_used.add(agent_label)

        criteria = self._merged_criteria(step)
        guidance = (
            "CitationPolicy:\n"
            "- 嚴禁使用 example.com/localhost/127.0.0.1 或占位連結。\n"
            "- 每則來源需提供 title/url/published_date（YYYY-MM-DD），url 必須可點且為原始來源（優先官網/原刊）。\n"
            "- 若為研究步驟，當達到 min_sources 且欄位齊全即可停止擴充（早停）。\n"
        )
        if attempt > 0:
            guidance += "- 前次未通過，請改用不同關鍵字/來源，並補齊缺欄位與有效連結。\n"

        input_payload = (
            f"Step: {step.description}\n"
            f"Params: {step.parameters}\n"
            f"AcceptanceCriteria: {json.dumps(criteria, ensure_ascii=False)}\n"
            f"{guidance}"
            f"Retry: {attempt + 1}/{total_attempts}"
        )

        task = Runner.run(agent, input_payload)
        timeout = self._cap_timeout(step)

        t0 = time.perf_counter()
        res = await asyncio.wait_for(task, timeout=timeout)
        dt = time.perf_counter() - t0

        self._agent_usage.append({
            "step_id": step.id,
            "step_num": self._step_order.get(step.id),
            "agent": agent_label,
            "seconds": round(dt, 2),
        })
        return str(res.final_output)

    def _route_agent(self, step: Step) -> Agent:
        if step.requires_tool and step.tool_name == "web_search":
            return research_agent
        if step.requires_tool and step.tool_name == "data_transform":
            return data_agent
        if step.requires_tool and step.tool_name == "code_run":
            return code_agent
        return research_agent

    def _merged_criteria(self, step: Step) -> Dict:
        base = _ensure_dict(step.acceptance_criteria)
        typ = base.get("type")
        # 統一補上防呆規則：研究類加欄位與禁用域名
        if typ == "research":
            base.setdefault("per_source_fields", ["title", "url", "published_date", "summary"])
            base.setdefault("forbid_domains", ["example.com", "localhost", "127.0.0.1"])
            # 若 planner 沒給 min_sources，給個合理下限
            base.setdefault("min_sources", 4)
        return base

# Chat 狀態
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "嗨嗨～請描述你的目標或要解的問題，安妮亞會規劃→並行研究→彙整交付！🥜"}
    ]

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

prompt = st.chat_input("請輸入你的目標或要解的問題（可持續補充）", max_chars=2000, key="chat_input_main")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🤩"):
        st.markdown(prompt)

    full_text = transcript_from_messages(st.session_state.messages)

    with st.chat_message("assistant", avatar="🧠"):
        with st.status("準備中…", state="running") as status:
            # 一個「過程紀錄」總 expander
            proc = st.expander("過程紀錄（即時更新）", expanded=True)
            triage_box = proc.container()
            plan_box = proc.container()
            steps_box = proc.container()
            writer_box = proc.container()
            verify_box = proc.container()
            agents_box = proc.container()

            def _progress_cb(event: str, info: Dict):
                # 狀態列
                if event == "triage.start":
                    status.update(label="分類中（Triage）", state="running")
                elif event == "triage.done":
                    status.update(label="分類完成（Triage）", state="running")
                    tri = info.get("triage")
                    if tri:
                        with triage_box:
                            st.markdown("• Triage 摘要")
                            st.markdown(f"- category: {tri.category}")
                            st.markdown(f"- complexity: {tri.complexity}")
                            st.markdown(f"- approach: {tri.approach}")
                            if tri.recommended_tools:
                                st.markdown(f"- recommended_tools: {', '.join(tri.recommended_tools)}")
                            if tri.notes:
                                st.markdown(f"- notes: {tri.notes}")

                elif event == "plan.start":
                    status.update(label="規劃中（Plan）", state="running")
                elif event == "plan.done":
                    total = info.get("total_steps", 0)
                    status.update(label=f"規劃完成（{total} 步）", state="running")
                    with plan_box:
                        st.markdown(f"• 規劃完成（共 {total} 步）")

                elif event == "execute.start":
                    status.update(label="執行中（Execute）", state="running")
                elif event == "execute.batch_start":
                    batch = info.get("batch")
                    count = info.get("count")
                    with steps_box:
                        st.markdown(f"• 開始並行批次：{batch}（{count} 步）")
                elif event == "execute.step_start":
                    pass
                elif event in ("execute.step_ok", "execute.step_warn", "execute.step_error"):
                    step_num = info.get("step_num")
                    desc = info.get("desc")
                    out = info.get("output") or ""
                    tag = {"execute.step_ok": "✅ 完成", "execute.step_warn": "⚠️ 完成（警告）", "execute.step_error": "❌ 失敗"}[event]
                    with steps_box:
                        st.markdown(f"• 步驟 {step_num} {tag}：{desc}")
                        if out:
                            st.code(out[:1600] + ("..." if len(out) > 1600 else ""), language="markdown")
                elif event == "execute.done":
                    status.update(label="執行完成（Execute）", state="running")
                elif event == "write.start":
                    status.update(label="撰寫中（Write）", state="running")
                elif event == "write.done":
                    status.update(label="撰寫完成（Write）", state="running")
                    with writer_box:
                        st.markdown("• 撰寫完成（略顯示全文）")

                elif event == "final_verify.start":
                    status.update(label="驗證中（Final Verify）", state="running")
                elif event == "revise.start":
                    r = info.get("round")
                    issues = info.get("issues")
                    status.update(label=f"驗證未過，修訂第 {r} 回…", state="running")
                    with verify_box:
                        st.markdown(f"• 修訂第 {r} 回：根據以下問題修正")
                        st.code((issues or "")[:1600], language="markdown")
                elif event == "revise.done":
                    r = info.get("round")
                    passed = info.get("passed")
                    with verify_box:
                        st.markdown(f"• 修訂第 {r} 回完成 → {'✅ 通過' if passed else '仍未通過'}")
                elif event == "final_verify.done":
                    passed = info.get("passed")
                    if passed:
                        status.update(label="驗證完成（✅ 通過）", state="complete")
                    else:
                        issues = info.get("issues")
                        status.update(label="驗證完成（⚠️ 未通過）", state="error")
                        with verify_box:
                            st.markdown("• 驗證問題")
                            st.code((issues or "")[:2000], language="markdown")

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
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    out = loop.run_until_complete(_run_once())
                except Exception as e:
                    status.update(label=f"流程失敗：{e}", state="error")
                    st.error(f"流程失敗：{e}")
                    st.stop()
                finally:
                    loop.close()
            except Exception as e:
                status.update(label=f"流程失敗：{e}", state="error")
                st.error(f"流程失敗：{e}")
                st.stop()

        # Plan（以友善步驟編號，不顯示 S1/S2…）
        plan = out.get("plan")
        if plan:
            with st.expander("Plan 步驟（含並行標註）", expanded=False):
                for i, s in enumerate(plan.steps, start=1):
                    tag = "並行" if s.is_parallel else "序列"
                    tool = f"{s.tool_name}" if s.tool_name else "-"
                    st.markdown(f"**步驟 {i}** · {tag} · tool={tool}")
                    st.markdown(f"- {s.description}")

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

        # Agents 使用與提速觀察
        agents_used = out.get("agents_used") or []
        usage = out.get("agent_usage") or []
        if agents_used:
            with st.expander("Agents 使用與耗時觀察", expanded=False):
                st.markdown(f"- 使用的 Agents：{', '.join(agents_used)}")
                if usage:
                    by_agent: Dict[str, float] = {}
                    for rec in usage:
                        by_agent[rec["agent"]] = by_agent.get(rec["agent"], 0.0) + float(rec["seconds"])
                    st.markdown("- 耗時（秒）彙總：")
                    for k, v in by_agent.items():
                        st.markdown(f"  - {k}: {round(v, 2)}s")
                    st.caption("提速建議：提高 research 並行數到 5、每步 timeout 自動封頂（research≤90s），且重試僅 1 次；已啟用連結防呆與早停。")

        st.session_state.messages.append({"role": "assistant", "content": final_output or "(流程完成)（無最終輸出）"})
