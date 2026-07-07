# -*- coding: utf-8 -*-
"""跨 session 教訓筆記（Supabase anya_lessons 表）。

設計原則（對齊 memory 紀律）：
- 一課一列：summary 一行可執行摘要 + content 短說明。
- 寧可更新不要重複：save() 先做相似度查重（difflib，同 category 內 summary 相似 ≥ 0.72 → 更新該列）。
- 全程軟失敗：任何網路/設定錯誤回傳空值或 {"ok": False}，絕不讓主流程炸掉。
- v1 刻意不用 pgvector：教訓量級是「數十列」，difflib 查重 + 時間排序已足夠；
  之後要做語意檢索再加 embedding 欄位即可（表結構已預留 use_count）。

建表 SQL 見 sql/anya_lessons.sql（在 Supabase SQL Editor 執行一次）。
"""
from datetime import datetime, timezone
from difflib import SequenceMatcher

VALID_CATEGORIES = ("search_strategy", "user_pref", "domain", "workflow")
_SIMILAR_THRESHOLD = 0.72   # 同 category 內 summary 相似度 ≥ 此值視為同一課 → 更新不新增


class LessonsStore:
    """Supabase anya_lessons 表的最小讀寫介面（供 Anya_Gemma save_lesson 工具與開場注入使用）。"""

    def __init__(self, url: str, key: str):
        # 延遲 import：缺 supabase 套件時只讓 store 建立失敗（呼叫端 try/except 降級），不擋整頁
        from supabase import create_client
        self._client = create_client(url, key)
        self._table = "anya_lessons"

    def fetch(self, limit: int = 8) -> list:
        """取最近更新的 N 課（開場注入用）。失敗回空 list。"""
        try:
            res = (
                self._client.table(self._table)
                .select("id,category,summary,content,updated_at")
                .order("updated_at", desc=True)
                .limit(int(limit))
                .execute()
            )
            return list(res.data or [])
        except Exception:
            return []

    @staticmethod
    def _similar(a: str, b: str) -> float:
        return SequenceMatcher(None, (a or "").strip(), (b or "").strip()).ratio()

    def save(self, category: str, summary: str, content: str) -> dict:
        """查重後寫入：同 category 內 summary 相似 ≥ 0.72 → 更新該列，否則新增。
        回傳 {"ok": True, "action": "insert"|"update"} 或 {"ok": False, "error": str}。"""
        cat = (category or "").strip()
        if cat not in VALID_CATEGORIES:
            return {"ok": False, "error": f"category 必須是 {'/'.join(VALID_CATEGORIES)} 之一"}
        s = (summary or "").strip()[:200]
        c = (content or "").strip()[:2000] or s
        if not s:
            return {"ok": False, "error": "summary 不可為空"}
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            existing = (
                self._client.table(self._table)
                .select("id,summary")
                .eq("category", cat)
                .order("updated_at", desc=True)
                .limit(50)
                .execute()
                .data
                or []
            )
            match = None
            for row in existing:
                if self._similar(s, row.get("summary") or "") >= _SIMILAR_THRESHOLD:
                    match = row
                    break
            if match:
                self._client.table(self._table).update(
                    {"summary": s, "content": c, "updated_at": now_iso}
                ).eq("id", match["id"]).execute()
                return {"ok": True, "action": "update", "id": match["id"]}
            self._client.table(self._table).insert(
                {"category": cat, "summary": s, "content": c, "updated_at": now_iso}
            ).execute()
            return {"ok": True, "action": "insert"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
