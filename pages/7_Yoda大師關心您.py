# ====== 輸出格式檢查器：fortune_agent 產物驗證 + 自動重試 ======

_BANNED_STRINGS = [
    "出生地", "時區", "DST", "日光節約", "日光節約時間",
    # 你要求不要出現英文段標
    "INNER_SKY", "YESTERDAYS_SKY", "CHANGING_SKY",
]

# 你要求的三段中文段標（必須出現）
_REQUIRED_SECTIONS_HAS_CHART = [
    "你內在的核心劇本：",
    "你曾用來活下來的方式：",
    "你接下來更成熟的選擇：",
]

_REQUIRED_FIELDS_BASE = [
    "STATUS:",
    "CHART_TYPES:",
    "CONSULT_GOAL:",
    "CONSULT_FOCUS:",
]

_REQUIRED_FIELDS_HAS_CHART = [
    "THEME:",
    "SHADOW:",
    "GIFT:",
    "CHOICE:",
    "PRACTICE:",
    "ACTIONS:",
]

_REQUIRED_FIELDS_NO_CHART = [
    "REASON:",
    "THEME:",
    "SHADOW:",
    "GIFT:",
    "CHOICE:",
    "PRACTICE:",
]

def _normalize_fortune_block(text: str) -> str:
    """
    容忍 fortune_agent 可能輸出 [ FORTUNE_SUMMARY ] 這種帶空白版本，
    統一正規化成 [FORTUNE_SUMMARY] / [/FORTUNE_SUMMARY]。
    """
    if not text:
        return ""

    t = text.strip()

    # normalize open/close tags with optional spaces
    t = re.sub(r"\[\s*FORTUNE_SUMMARY\s*\]", "[FORTUNE_SUMMARY]", t)
    t = re.sub(r"\[\s*/\s*FORTUNE_SUMMARY\s*\]", "[/FORTUNE_SUMMARY]", t)
    t = re.sub(r"\[\s*FULL_CHART\s*\]", "[FULL_CHART]", t)
    t = re.sub(r"\[\s*/\s*FULL_CHART\s*\]", "[/FULL_CHART]", t)
    return t.strip()


def _extract_fortune_summary_block(text: str) -> Optional[str]:
    """
    只抽出 [FORTUNE_SUMMARY]...[/FORTUNE_SUMMARY] 本體；若不存在回傳 None。
    """
    t = _normalize_fortune_block(text)
    m = re.search(r"\[FORTUNE_SUMMARY\][\s\S]*?\[/FORTUNE_SUMMARY\]", t)
    if not m:
        return None
    return m.group(0).strip()


def _is_only_one_fortune_block(text: str) -> bool:
    """
    要求：整段輸出只能是 fortune block（前後不得有其他文字）。
    """
    t = _normalize_fortune_block(text)
    block = _extract_fortune_summary_block(t)
    if not block:
        return False
    return t == block


def _parse_status(block: str) -> Optional[str]:
    m = re.search(r"STATUS:\s*(HAS_CHART|NO_CHART)\b", block)
    return m.group(1) if m else None


def _validate_fortune_output(raw_text: str) -> Tuple[bool, List[str], Optional[str]]:
    """
    回傳 (ok, problems, normalized_block)
    - ok: 是否符合格式
    - problems: 不符合原因列表（用於重試提示）
    - normalized_block: 正規化後的 fortune block（通過才有意義）
    """
    problems: List[str] = []
    t = _normalize_fortune_block(raw_text)

    # 1) 必須存在且只能有一個 block
    block = _extract_fortune_summary_block(t)
    if not block:
        problems.append("缺少 [FORTUNE_SUMMARY]...[/FORTUNE_SUMMARY] 區塊")
        return False, problems, None
    if not _is_only_one_fortune_block(t):
        problems.append("輸出包含 fortune 區塊以外的多餘文字（必須只輸出 fortune 區塊）")

    # 2) 禁詞
    for s in _BANNED_STRINGS:
        if s in block:
            problems.append(f"包含禁詞/禁段標：{s}")

    # 3) 必要欄位（共同）
    for key in _REQUIRED_FIELDS_BASE:
        if key not in block:
            problems.append(f"缺少欄位：{key}")

    status = _parse_status(block)
    if status is None:
        problems.append("STATUS 必須是 HAS_CHART 或 NO_CHART")
        return False, problems, block

    # 4) 依 status 檢查
    if status == "HAS_CHART":
        for sec in _REQUIRED_SECTIONS_HAS_CHART:
            if sec not in block:
                problems.append(f"HAS_CHART 缺少中文段落標題：{sec}")

        for key in _REQUIRED_FIELDS_HAS_CHART:
            if key not in block:
                problems.append(f"HAS_CHART 缺少欄位：{key}")

        # ACTIONS 至少要有一條 - 1) ... 形式
        if "ACTIONS:" in block and not re.search(r"ACTIONS:\s*\n-\s*1\)", block):
            problems.append("ACTIONS 需包含至少一條條列（例如 '- 1) ...'）")

    else:  # NO_CHART
        for key in _REQUIRED_FIELDS_NO_CHART:
            if key not in block:
                problems.append(f"NO_CHART 缺少欄位：{key}")

    ok = len(problems) == 0
    return ok, problems, block


async def _run_fortune_checked(
    user_id: str,
    system_info: str,
    user_message: str,
    session: EncryptedSession,
    max_attempts: int = 2,
) -> Optional[str]:
    """
    會自動重試 fortune_agent，直到通過格式檢查或耗盡嘗試次數。
    """
    last_block: Optional[str] = None
    last_problems: List[str] = []

    for attempt in range(1, max_attempts + 1):
        format_hint = ""
        if attempt > 1 and last_problems:
            # 這段只給 fortune_agent 看的，不會直接給使用者
            format_hint = (
                "[FORMAT_HINT]\n"
                "上一次輸出未通過格式檢查，請你這次務必完全修正。\n"
                "問題如下（請逐一修正）：\n"
                + "\n".join([f"- {p}" for p in last_problems])
                + "\n要求：只能輸出一個 [FORTUNE_SUMMARY] 區塊，且使用中文三段標題。\n"
                "[/FORMAT_HINT]\n"
            )

        full_input = system_info + format_hint + f"[USER MESSAGE] {user_message}"
        r = await Runner.run(fortune_agent, input=full_input, session=session)
        raw = (r.final_output or "").strip()

        ok, problems, block = _validate_fortune_output(raw)
        last_problems = problems
        last_block = block

        if ok and block:
            return block

    # 失敗就回傳「最後一次抽到的 block」（可能讓 counselor 至少能做點事），但不進快取
    return last_block
