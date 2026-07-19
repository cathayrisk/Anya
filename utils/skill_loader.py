# -*- coding: utf-8 -*-
"""skills/ 與 .claude/agents/ 掃描器（Anya_Gemma load_skill / consult_expert 的外部來源）。

設計原則：
- 白名單制：skills/ 底下有 35+ 個資料夾，全上會灌爆 system prompt 索引，
  只註冊下方 SKILL_WHITELIST / AGENT_WHITELIST 挑過的；要加新項目改白名單即可。
- 索引描述用中文短句（給 Gemma 的觸發線索），不用 SKILL.md 裡動輒百字的英文 description。
- 惰性載入：discover 階段只記路徑不讀內容，load_skill / consult_expert 被呼叫時才讀檔，
  避免啟動時把 30 份 SKILL.md 全塞記憶體。
- 全程軟失敗：掃不到、讀不到就跳過該條目，絕不讓主頁面炸掉。
"""
import os
import re

# 專案根目錄（utils/ 的上一層）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== Skill 白名單：{SKILL.md frontmatter name: 中文索引描述} =====
# 名稱必須和 SKILL.md frontmatter 的 name 完全一致（掃描時以此配對）。
SKILL_WHITELIST: dict[str, str] = {
    "caveman": "極簡回覆模式——砍掉贅字客套、保留完整技術正確性（使用者要求「簡短點」「省 token」時）",
    "reflect": "對話中途退一步重估——方向/假設/偏誤五維檢查（使用者說「等等」「我們是不是想偏了」時）",
    "roast": "誠實吐槽模式——直白指出計畫/程式碼/文案的問題，不留情面（使用者要求誠實評價時）",
    "market-research": "市場研究方法論——TAM/SAM/SOM 雙向估算、問卷樣本數、市場區隔評分",
    "product-research": "產品研究方法論——用戶訪談設計、假設驗證、機會評估",
    "statistical-analyst": "統計分析——假設檢定、信賴區間、迴歸、效果量的正確用法與陷阱",
    "financial-analyst": "財務分析——DCF 估值、財報比率、預算預測、SaaS 指標（ARR/CAC/LTV）",
    "data-quality-auditor": "資料品質稽核——缺值/重複/離群/型別不一致的系統性檢查清單",
    "universal-scraping-architect": "網頁爬取架構——反爬因應、選擇器策略、結構變動的穩健設計",
    "karpathy-coder": "Karpathy 風格寫碼紀律——最小可行改動、先讀後寫、避免過度工程",
    "md-document": "Markdown 轉精美 HTML 文件——排版/目錄/樣式模板",
    "landing": "Landing page HTML 產生器——4 種設計風格可選",
    # 原有本地 skills（python_best_practices 已由寫死條目涵蓋；
    # supabase-postgres-best-practices 是目錄型、規則本文缺檔，刻意不收）
    "security-checklist": "程式碼安全檢查——涉及 secrets/資料庫/檔案 IO/網路請求/使用者輸入時必查",
    "sql-and-database": "SQL 與資料庫實務——SQLAlchemy/原生 SQL/交易/連線管理/migration",
}

# ===== Agent 白名單：{consult_expert role key: 條目設定} =====
# path 相對於專案根目錄；skill 選填 = 派工時自動前置該 skill 的方法論全文。
AGENT_WHITELIST: dict[str, dict] = {
    "financial_analyst": {
        "label": "💰 財務分析師",
        "description": "DCF 估值、財務建模、預算預測、SaaS 指標健檢",
        "path": os.path.join(".claude", "agents", "finance", "cs-financial-analyst.md"),
        "skill": "financial-analyst",
    },
    "product_analyst": {
        "label": "📊 產品分析師",
        "description": "KPI 設計、實驗設計、產品數據分析與機會評估",
        "path": os.path.join(".claude", "agents", "product", "cs-product-analyst.md"),
        "skill": "product-research",
    },
    "content_creator": {
        "label": "✍️ 內容創作者",
        "description": "品牌語氣一致的長短文寫作與內容企劃",
        "path": os.path.join(".claude", "agents", "marketing", "cs-content-creator.md"),
        "skill": None,
    },
}

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_NAME_RE = re.compile(r"^name:\s*['\"]?([\w./-]+)['\"]?\s*$", re.MULTILINE)

# 掃描到的 persona 是英文寫成，補一行輸出語言規則（既有中文 persona 不受影響）
_ZH_SUFFIX = "\n\n---\n除非任務內容另有指定，一律以繁體中文回覆。"


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _frontmatter_name(text: str) -> str:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return ""
    nm = _NAME_RE.search(m.group(1))
    return nm.group(1).strip() if nm else ""


def strip_frontmatter(text: str) -> str:
    """去掉 YAML frontmatter（persona/skill 本文才是要餵給模型的部分）。"""
    return _FRONTMATTER_RE.sub("", text, count=1)


def discover_skills(root: str = _ROOT) -> dict[str, dict]:
    """掃 skills/ 樹找白名單內的 SKILL.md → {name: {description, path, content: None}}。

    同時涵蓋兩種 layout：skills/<x>/SKILL.md 與 skills/<x>/skills/<y>/SKILL.md
    （上游 repo 的 plugin 包裝）。以 frontmatter name 配對白名單，路徑不重要。
    """
    found: dict[str, dict] = {}
    skills_dir = os.path.join(root, "skills")
    if not os.path.isdir(skills_dir):
        return found
    for dirpath, dirnames, filenames in os.walk(skills_dir):
        # 深度限制：skills/ 起算最多 4 層（防呆，實際 layout 都在 3 層內）
        depth = os.path.relpath(dirpath, skills_dir).count(os.sep)
        if depth >= 4:
            dirnames[:] = []
            continue
        if "SKILL.md" not in filenames:
            continue
        path = os.path.join(dirpath, "SKILL.md")
        try:
            name = _frontmatter_name(_read(path))
        except Exception:
            continue
        if name in SKILL_WHITELIST and name not in found:
            found[name] = {
                "description": SKILL_WHITELIST[name],
                "path": path,
                "content": None,   # 惰性：load_skill 時才讀
            }
    return found


def load_skill_content(entry: dict) -> str:
    """讀取 skill 全文（快取回 entry["content"]）。失敗回空字串。"""
    if entry.get("content"):
        return entry["content"]
    path = entry.get("path")
    if not path:
        return ""
    try:
        entry["content"] = strip_frontmatter(_read(path)).strip()
    except Exception:
        return ""
    return entry["content"]


def discover_agents(root: str = _ROOT) -> dict[str, dict]:
    """載入白名單內的 persona 檔 → CONSULT_ROLES 相容格式
    {key: {label, description, prompt: None, prompt_path, skill}}。prompt 同樣惰性載入。"""
    found: dict[str, dict] = {}
    for key, cfg in AGENT_WHITELIST.items():
        path = os.path.join(root, cfg["path"])
        if not os.path.isfile(path):
            continue
        found[key] = {
            "label": cfg["label"],
            "description": cfg["description"],
            "prompt": None,        # 惰性：consult_expert 時才讀
            "prompt_path": path,
            "skill": cfg.get("skill"),
        }
    return found


def resolve_role_prompt(role_entry: dict, skills: dict) -> str:
    """組出 subagent 的 persona prompt：persona 本文（＋配對 skill 方法論全文）。

    - 既有寫死角色（prompt 已是字串）原樣回傳，行為零改變。
    - 掃描角色：讀 prompt_path、去 frontmatter、補中文輸出規則；
      有配對 skill 且載得到時，把方法論全文接在 persona 後面（單發呼叫的組合技）。
    """
    if role_entry.get("prompt"):
        return role_entry["prompt"]
    path = role_entry.get("prompt_path")
    if not path:
        return ""
    try:
        persona = strip_frontmatter(_read(path)).strip()
    except Exception:
        return ""
    skill_name = role_entry.get("skill")
    if skill_name and skill_name in skills:
        methodology = load_skill_content(skills[skill_name])
        if methodology:
            persona += f"\n\n## 作業方法論（必須遵循）\n\n{methodology}"
    persona += _ZH_SUFFIX
    role_entry["prompt"] = persona   # 快取：同 session 重複諮詢不重讀
    return persona
