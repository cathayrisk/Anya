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


# ── 400 INVALID_ARGUMENT（2026-09-05 T15）─────────────────────────────────────
# 實際崩潰時的原文：整輪丟失、畫面印 traceback，而當下 3.5-flash-lite 明明可用
MSG_400 = ("Error calling model 'gemma-4-31b-it' (INVALID_ARGUMENT): 400 INVALID_ARGUMENT. "
           "{'error': {'code': 400, 'message': 'Request contains an invalid argument.', "
           "'status': 'INVALID_ARGUMENT'}}")


class GoogleInvalidRequestError(Exception):
    pass


def test_400_is_classified_as_bad_request():
    """修之前四個旗標全 False、retriable 也 False → invoke_with_backoff 直接往上拋。"""
    k = classify_llm_error(GoogleInvalidRequestError(MSG_400))
    assert k.is_bad_request


def test_400_does_not_mark_the_model_dead():
    """is_dead 是永久的（_mark_model_dead 整個 session 跳過那一格）。
    但 400 是「這個請求」被拒，不是模型壞掉——下個回合換個請求它就正常了。"""
    k = classify_llm_error(GoogleInvalidRequestError(MSG_400))
    assert not k.is_dead


def test_400_is_not_retriable_on_the_same_model():
    """請求本身被拒，同一顆重打幾次都一樣。換模型的分支排在 `if not retriable: raise` 之前。"""
    k = classify_llm_error(GoogleInvalidRequestError(MSG_400))
    assert not k.retriable
    assert not (k.is_quota or k.is_stuck or k.is_overloaded)


@pytest.mark.parametrize("exc", [
    GoogleInvalidRequestError("InvalidArgument: bad parts"),
    Exception("google.api_core.exceptions.InvalidArgument"),
    type("BadRequestError", (Exception,), {})("something"),
])
def test_other_shapes_of_the_same_error_are_caught(exc):
    assert classify_llm_error(exc).is_bad_request


def test_does_not_false_positive_on_the_bare_number_400():
    """⚠️ 刻意不用裸的 "400" in msg：那會誤抓 token 數這類字串。
    誤判的代價是好好的回合被無謂降級到弱模型。"""
    assert not classify_llm_error(ValueError("prompt is 4000 tokens, limit 16000")).is_bad_request
    assert not classify_llm_error(ValueError("finished in 400 ms")).is_bad_request


def test_existing_errors_are_not_reclassified_as_bad_request():
    for msg in (MSG_503, MSG_429, MSG_404, MSG_500):
        assert not classify_llm_error(ServerError(msg)).is_bad_request, msg


def test_home_switches_model_on_400_without_marking_dead():
    """分類寫了但沒接進 Home.py 等於沒修——這正是 T15 的教訓。"""
    src = (ROOT / "Home.py").read_text(encoding="utf-8")
    i = src.index("if purpose and _k.is_bad_request:")
    region = src[i: i + 500]
    assert "downgrade_model(purpose)" in region
    assert "_mark_model_dead" not in region, "400 不可標記死亡，會白白毀掉一整格"
    # 必須排在 `if not retriable: raise` 之前，否則永遠走不到
    assert i < src.index("if not retriable:")
    assert "is_bad_request=_k.is_bad_request" in src, "telemetry 沒記就看不到之後發生幾次"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
