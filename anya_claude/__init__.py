# -*- coding: utf-8 -*-
"""Anya 的 Claude 版主聊天（pages/0_Anya_Claude.py 用）。

分層規則（spike T1 實測的結論，見 tools/claude_spike/README.md）：
LangGraph 在 token 串流模式下，節點會跑在背景執行緒、沒有 ScriptRunContext——
所以 graph.py／events.py／router.py 等**一律不 import streamlit**，畫面只在
view.py 與頁面本身（主執行緒）處理。tests/test_claude_layering.py 守著這條線。
"""
