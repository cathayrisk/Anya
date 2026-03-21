# Deep_Research/tools.py
# -*- coding: utf-8 -*-
"""
LangChain 工具供 DeepAgents 使用：
  - openai_web_search: 使用 OpenAI Responses API 搜尋網路
  - think: 策略性反思工具（來自 Anya_Test.py THINK_TOOL 模式）

doc_search 工具因需存取 Streamlit session_state，在頁面中動態建立。
"""

import os
from langchain_core.tools import tool
from openai import OpenAI


@tool
def openai_web_search(query: str) -> str:
    """使用 OpenAI web search 搜尋最新網路資訊，回傳摘要內容與來源 URL。
    適合搜尋近期事件、特定主題的公開資訊、新聞與研究報告。"""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            tools=[{"type": "web_search_preview"}],
            input=query,
        )
    except Exception as e:
        return f"搜尋失敗：{e}"

    text_parts: list[str] = []
    urls: list[str] = []

    for item in response.output:
        if not hasattr(item, "content"):
            continue
        for c in item.content:
            if hasattr(c, "text"):
                text_parts.append(c.text)
            if hasattr(c, "annotations"):
                for ann in c.annotations:
                    if hasattr(ann, "url"):
                        title = getattr(ann, "title", ann.url) or ann.url
                        urls.append(f"- [{title}]({ann.url})")

    result = "\n".join(text_parts)
    if urls:
        result += "\n\n**來源：**\n" + "\n".join(urls)
    return result[:4000] if result else "（無搜尋結果）"


@tool
def think(reflection: str, key_finding: str, next_action: str, confidence: int) -> str:
    """在工具呼叫之間進行策略性反思：分析研究進度、評估資訊品質、規劃下一步。
    此工具不會取得新資訊，僅記錄思考過程以提升研究品質。

    參數：
      reflection: 五面向反思（發現摘要、假設對比、矛盾偵測、資訊缺口、策略方向）
      key_finding: 本輪最重要的發現（1–2 句）
      next_action: 下一步策略，必須為 '繼續搜尋'、'換工具' 或 '直接作答' 之一
      confidence: 目前能完整回答問題的程度（0–100）
    """
    return (
        f"[Think ✓] confidence={confidence}% | "
        f"next={next_action} | "
        f"finding={key_finding[:100]}"
    )
