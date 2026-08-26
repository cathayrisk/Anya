# -*- coding: utf-8 -*-
"""列出這把 Google API 金鑰實際支援哪些模型，用來填 Home.py 的 MODEL_CHAINS。

為什麼需要這支：MODEL_CHAINS 裡填錯的模型 ID 會在執行期才失敗（雖然會被
_mark_model_dead 標記跳過，但每個錯 ID 仍會浪費一次呼叫）。先用這支確認再填。

用法（金鑰來源與 Home.py 相同的優先序：環境變數 → .streamlit/secrets.toml → .env）：
    python tools/list-google-models.py
    python tools/list-google-models.py --all      # 連不能對話的模型也列出
"""
from __future__ import annotations

import os
import re
import sys
import json
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    for path, pat in (
        (os.path.join(ROOT, ".streamlit", "secrets.toml"),
         re.compile(r'^\s*GOOGLE_API_KEY\s*=\s*["\']([^"\']+)["\']', re.M)),
        (os.path.join(ROOT, ".env"),
         re.compile(r'^\s*GOOGLE_API_KEY\s*=\s*["\']?([^"\'\s]+)', re.M)),
    ):
        try:
            with open(path, encoding="utf-8") as f:
                m = pat.search(f.read())
            if m:
                return m.group(1)
        except OSError:
            continue
    return ""


def main() -> int:
    key = _find_key()
    if not key:
        print("找不到 GOOGLE_API_KEY（環境變數 / .streamlit/secrets.toml / .env 都沒有）")
        return 1

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=200"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"查詢失敗：{type(e).__name__}: {e}")
        return 1

    show_all = "--all" in sys.argv
    rows = []
    for m in data.get("models", []):
        name = (m.get("name") or "").replace("models/", "")
        methods = m.get("supportedGenerationMethods") or []
        chat_ok = "generateContent" in methods
        if not chat_ok and not show_all:
            continue
        rows.append((name, m.get("inputTokenLimit"), m.get("outputTokenLimit"), methods))

    rows.sort()
    print(f"共 {len(rows)} 個模型" + ("" if show_all else "（僅列可對話的）") + "\n")
    fam: dict[str, list] = {}
    for name, tin, tout, _ in rows:
        key_fam = "gemma" if name.startswith("gemma") else name.split("-")[0]
        fam.setdefault(key_fam, []).append((name, tin, tout))
    for k in sorted(fam):
        print(f"── {k} ──")
        for name, tin, tout in fam[k]:
            print(f"  {name:<44} in={tin!s:<9} out={tout}")
        print()

    print("填進 Home.py 的 MODEL_CHAINS 時要注意：")
    print("  1. 備援模型必須支援 function calling，否則 General 的工具迴圈會壞掉")
    print("  2. 同家族（gemma↔gemma、gemini↔gemini）的回答風格較接近，降級比較不突兀")
    print("  3. 免費層的每日／每分鐘限制各模型不同，別把稀缺模型放進高頻用途")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
