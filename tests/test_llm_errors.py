# -*- coding: utf-8 -*-
"""503 換模型（utils/llm_errors.py）。

2026-09-04 A/B 實測：gemini-3.5-flash-lite 連續 21 次 503 跨 8 分鐘。舊版 invoke_with_backoff
只把 503 當 retriable、不 downgrade_model()，fast 鏈第二格正是 3.5-lite → 一降級就卡死。
這組測試守：(1) 分類純函式對真實錯誤字串的判定；(2) Home.py 的降級條件真的把 is_overloaded 接進去。

跑法：python -m pytest tests/test_llm_errors.py -v
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "Home.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.llm_errors import classify_llm_error  # noqa: E402

# 2026-09-04 tools/ab-flash-lite.py 抓到的原文（金鑰已遮罩）
MSG_503 = ("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing "
           "high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}")
MSG_429 = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, "
           "please check your plan and billing details.', 'status': 'RESOURCE_EXHAUSTED'}}")
MSG_404 = "404 NOT_FOUND. {'error': {'code': 404, 'message': 'models/gemini-2.5-flash-lite is not found'}}"
MSG_500 = "500 INTERNAL. {'error': {'code': 500, 'message': 'An internal error has occurred.'}}"


class ServerError(Exception):
    pass


class ClientError(Exception):
    pass


class ReadTimeout(Exception):
    pass


def test_503_is_overloaded_not_quota_not_stuck():
    k = classify_llm_error(ServerError(MSG_503))
    assert k.is_overloaded and k.retriable
    assert not k.is_quota and not k.is_stuck and not k.is_dead


def test_429_is_quota():
    k = classify_llm_error(ClientError(MSG_429))
    assert k.is_quota and k.retriable and not k.is_overloaded


def test_timeout_is_stuck():
    assert classify_llm_error(ReadTimeout("The read operation timed out")).is_stuck
    assert classify_llm_error(TimeoutError()).is_stuck


def test_404_is_dead():
    assert classify_llm_error(ClientError(MSG_404)).is_dead


def test_500_retriable_but_not_overloaded():
    """gemma 池的 500 是單次抖動，同顆短退避即可，不該降級掉品質（設計決定，見模組 docstring）。"""
    k = classify_llm_error(ServerError(MSG_500))
    assert k.retriable and not k.is_overloaded


def test_unknown_error_not_retriable():
    k = classify_llm_error(ValueError("boom"))
    assert not (k.retriable or k.is_quota or k.is_stuck or k.is_overloaded or k.is_dead)


def test_home_wires_overloaded_into_downgrade():
    """分類寫了但沒接進降級條件等於沒修。"""
    src = (ROOT / "Home.py").read_text(encoding="utf-8")
    assert "from utils.llm_errors import classify_llm_error" in src
    assert "(is_quota or is_stuck or is_overloaded) and purpose and downgrade_model(purpose)" in src
    assert "is_overloaded=is_overloaded" in src, "telemetry 沒記 is_overloaded，之後就看不到 503 降級了幾次"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
