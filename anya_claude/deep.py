# -*- coding: utf-8 -*-
"""階段 2：深思模式改用 deepagents（create_deep_agent）。純 Python，無 streamlit 依賴。

組成：
- 模型：Opus 5.5（effort medium）＋ server tools（基本版 web_search_20250305、web_fetch_20250910——
  動態篩選版會在沙盒程式裡呼叫自訂工具，續跑會 400，見 config.DEEP_SERVER_TOOLS）
- 自訂工具：氣象署三支、think（anya_claude/tools.py）
- skills：照 utils/skill_loader.SKILL_WHITELIST 的 21 個，描述換成中文，用 StateBackend 的虛擬檔案
  放在 /skills/<名稱>/（spike T4B 驗過：Opus 會自己 read_file 讀 SKILL.md）
- middleware：pause_turn 續跑、用量事件（deep_mw.py）；快取 TTL 改 1h（見下）
- 關掉 deepagents 自動加的「通用子代理」（task 工具會自己再開一個 Opus 跑整題，成本難控；
  計畫的子代理在階段 4 才做，屆時用指定角色）

快取 TTL：deepagents 0.7.30 一定會加 AnthropicPromptCachingMiddleware（預設 5 分鐘），使用者無法指定。
同事常隔超過 5 分鐘才問下一題，5 分鐘快取幾乎每題都重寫。這裡用「同名子類、預設 1h」換掉
deepagents 內部引用的那個類別（private 模組，所以釘版並有測試盯著）。
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware as _BaseCaching

from . import config as CFG
from . import prompts as P
from .deep_mw import (PauseTurnMiddleware, PrevRequestBreakpointMiddleware, RubricStartMiddleware,
                      UsageEventsMiddleware, grader_view, make_deliverable_rubric, rubric_event)

RUBRIC_MAX_ITERATIONS = 2   # 評兩次＝最多補寫一次（第二次還不過就照現況交出去，時間軸會標出來）
from .tools import CWA_TOOLS, calculate, create_widget, think

ROOT = Path(__file__).resolve().parents[1]
_SKILL_SUFFIXES = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".bas", ".csv"}


MESSAGES_TTL = "5m"


class AnthropicPromptCachingMiddleware(_BaseCaching):
    """同名（deepagents 用 middleware 的 name 辨識）。系統提示與工具用 1h，對話部分用 5m：

    - 系統提示＋工具＋skills 索引（約 11K token）每題都一樣 → 1h，同事隔很久再問也讀得到
    - 對話部分（這回合的搜尋結果、工具結果）只在同一回合的下一次模型呼叫用到；下一回合的歷史是純文字重建，
      用不到這些區塊 → 寫 5m 就好（寫入 $5 vs 1h 的 $8／MTok）。API 規定長 TTL 要在短 TTL 前面，這個順序符合。
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ttl", "1h")
        super().__init__(*args, **kwargs)

    def _apply_caching(self, request):
        req = super()._apply_caching(request)
        return req.override(model_settings={**req.model_settings,
                                            "cache_control": {"type": "ephemeral", "ttl": MESSAGES_TTL}})


def install_cache_ttl() -> None:
    import deepagents.middleware._prompt_caching as dpc
    dpc.AnthropicPromptCachingMiddleware = AnthropicPromptCachingMiddleware


def install_opus_profile() -> None:
    from deepagents import HarnessProfile, register_harness_profile
    from deepagents.profiles import GeneralPurposeSubagentProfile
    register_harness_profile(f"anthropic:{CFG.OPUS}", HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)))


@lru_cache(maxsize=1)
def skill_files() -> dict[str, dict]:
    """白名單 skills → deepagents 虛擬檔案 {"/skills/<name>/<相對路徑>": file_data}。

    SKILL.md 的 description 換成白名單的中文短句（deepagents 把它放進系統提示當索引；中文的觸發線索比原文準）。
    """
    from deepagents.backends.utils import create_file_data
    from utils.skill_loader import SKILL_WHITELIST, discover_skills

    files: dict[str, dict] = {}
    for name, entry in discover_skills(str(ROOT)).items():
        d = Path(entry["path"]).parent
        for f in sorted(d.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in _SKILL_SUFFIXES:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            if f == Path(entry["path"]):
                text = re.sub(r"^description:.*?(?=^\w[\w-]*:|^---)", f"description: {SKILL_WHITELIST[name]}\n",
                              text, count=1, flags=re.M | re.S)
                text = re.sub(r"^name:.*$", f"name: {name}", text, count=1, flags=re.M)
            files[f"/skills/{name}/{f.relative_to(d).as_posix()}"] = create_file_data(text)
    # 互動元件模板（widget_templates.py，舊版用 load_skill 載入）也放成 skill；名稱改連字號（Agent Skills 規格）
    from widget_templates import WIDGET_TEMPLATES
    for key, tpl in WIDGET_TEMPLATES.items():
        name = key.replace("_", "-")
        files[f"/skills/{name}/SKILL.md"] = create_file_data(
            f"---\nname: {name}\ndescription: {tpl['description']}\n---\n\n{tpl['content']}")
    return files


def build_deep_agent(*, socratic: bool = False):
    from deepagents import create_deep_agent

    from . import astro_tools, lessons

    install_cache_ttl()
    install_opus_profile()
    model = ChatAnthropic(model=CFG.OPUS, max_tokens=CFG.OPUS_MAX_TOKENS, effort=CFG.OPUS_EFFORT)
    # 長期記憶只在有設定 Supabase 時才給；占星只在裝了 kerykeion 時才給（舊版也是這樣降級）
    extra = [lessons.save_lesson] if lessons.enabled() else []
    extra += astro_tools.ASTRO_TOOLS if astro_tools.available() else []
    middleware = [PauseTurnMiddleware(), UsageEventsMiddleware(label="Opus 深思", model_id=CFG.OPUS),
                  PrevRequestBreakpointMiddleware()]
    if not socratic:   # 提問引導模式是在問問題、不是交答案，通則評分不適用
        middleware += [_rubric_middleware(), RubricStartMiddleware()]
    return create_deep_agent(
        model=model,
        tools=[*CWA_TOOLS, think, calculate, create_widget, *extra, *CFG.DEEP_SERVER_TOOLS],
        system_prompt=P.deep_system(socratic=socratic),
        skills=["/skills/"],
        middleware=middleware,
        name="anya-deep",
    )


def _rubric_middleware():
    """寫完再檢查：Haiku 對照 prompts.DEEP_RUBRIC 打分，沒過就把缺口回饋給 Opus 補寫（最多一次）。
    只在長交付物上跑（讀了 skill、回答有程式碼、報告型長回答——deep_mw.deliverable_reason；
    2026-10-23 使用者決定一般問答不要跑：Haiku 評分不一致，一般問答每題多 15–20 秒卻幾乎每次都挑毛病）。
    呼叫時 state 也要有 rubric（graph.opus_node 決定，提問引導模式不放）。spike T6 驗過機制。"""
    from deepagents.middleware.rubric import RubricMiddleware
    grader = ChatAnthropic(model=CFG.HAIKU, max_tokens=4000, effort="low", thinking=CFG.HAIKU_THINKING)
    return make_deliverable_rubric(RubricMiddleware)(
        model=grader, max_iterations=RUBRIC_MAX_ITERATIONS, on_evaluation=rubric_event,
                            system_prompt=grader_prompt(),
                            prepare_messages_for_grader=grader_view)


def grader_prompt() -> str:
    """deepagents 的評分提示，把「無法確認就判失敗」換成「有明確缺陷才判失敗」（理由見 prompts.GRADER_LENIENT）。
    內建提示改了字就找不到那一句——那時改成附在最後，tests 也會紅。"""
    from deepagents.middleware.rubric import GRADER_SYSTEM_PROMPT
    if P.GRADER_CONSERVATIVE in GRADER_SYSTEM_PROMPT:
        base = GRADER_SYSTEM_PROMPT.replace(P.GRADER_CONSERVATIVE, P.GRADER_LENIENT)
    else:
        base = GRADER_SYSTEM_PROMPT + "\n\n" + P.GRADER_LENIENT
    return base + P.GRADER_LANGUAGE_NOTE
