# Deep_Research/openai_retriever.py
# -*- coding: utf-8 -*-
"""
gpt-researcher custom retriever — 使用 OpenAI Responses API web_search_preview
不需要 Tavily API Key。

gpt-researcher 期望的介面：
  - __init__(self, query: str)
  - search(self, max_results: int) -> list[dict]
      每個 dict 格式：{"href": url, "body": text, "title": title}
"""

import os
from openai import OpenAI


class OpenAIWebRetriever:
    """gpt-researcher custom retriever，使用 OpenAI Responses API web_search_preview。"""

    def __init__(self, query: str):
        self.query = query
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def search(self, max_results: int = 5) -> list[dict]:
        """執行 web search 並回傳 gpt-researcher 期望的格式。"""
        try:
            response = self._client.responses.create(
                model="gpt-4.1-mini",
                tools=[{"type": "web_search_preview"}],
                input=self.query,
            )
        except Exception:
            return []

        results = []
        for item in response.output:
            if not hasattr(item, "content"):
                continue
            for c in item.content:
                if hasattr(c, "annotations"):
                    for ann in c.annotations:
                        if hasattr(ann, "url"):
                            results.append({
                                "href": ann.url,
                                "body": getattr(ann, "title", "") or self.query,
                                "title": getattr(ann, "title", ann.url),
                            })

        # 若 annotation 沒有 URL，把文字結果整包回傳
        if not results:
            text = "".join(
                c.text
                for item in response.output
                if hasattr(item, "content")
                for c in item.content
                if hasattr(c, "text")
            )
            if text:
                results.append({
                    "href": "",
                    "body": text[:3000],
                    "title": self.query,
                })

        return results[:max_results]
