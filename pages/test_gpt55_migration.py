"""
GPT-5.5 遷移驗證腳本（commit 前手動跑）

用法：
    cd <project root>
    python test_gpt55_migration.py

需要：OPENAI_API_KEY 已在環境變數或 .streamlit/secrets.toml
"""
import os
import sys
import time

# 從 secrets.toml 讀 key（與 Home.py 一致）
def load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        with open(secrets_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("OPENAI_API_KEY") or data.get("OPENAI_KEY")
    return None

key = load_key()
if not key:
    print("[FAIL] 找不到 OPENAI_API_KEY")
    sys.exit(1)
os.environ["OPENAI_API_KEY"] = key

sys.path.insert(0, ".")

# ── 測試 1：gpt-5.5 連線 ─────────────────────────────────────────────────────
print("\n=== 測試 1：gpt-5.5 連線 ===")
from openai import OpenAI
client = OpenAI()
try:
    t0 = time.time()
    resp = client.responses.create(
        model="gpt-5.5",
        input="說『你好』兩個字就好",
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )
    elapsed = time.time() - t0
    print(f"[PASS] gpt-5.5 可呼叫，回應：{resp.output_text!r}（{elapsed:.1f}s）")
except Exception as e:
    print(f"[FAIL] gpt-5.5 呼叫失敗：{type(e).__name__}: {e}")
    sys.exit(1)

# ── 測試 2：新規則 5（論證邏輯有效性）─────────────────────────────────────
print("\n=== 測試 2：規則 5 邏輯缺口偵測 ===")
# 這個測試報告刻意埋了三個邏輯問題：
# (a) 內部矛盾：第一段說獲利成長 35%，第二段說 EPS 下滑 10%
# (b) 邏輯跳躍：「Azure 成長 → 因此整體 AI 趨勢看漲」沒有中間步驟
# (c) 虛假兩難：只給「買入或賣出」沒考慮持有
buggy_report = """
微軟 Q3 財報：營收年增 13%，Azure 雲端營收年增 35%，獲利年增 35%。
因此 AI 趨勢確認看漲，整體市場應全面布局 AI 概念股。
EPS 從 14.92 下滑到 14.31，年減 10%。
投資人現在只有兩個選擇：買入或賣出，沒有持有的空間。
建議買入評級。
"""
from cowork.critic_pipeline import run_critic_pipeline
try:
    result = run_critic_pipeline(buggy_report)
    print(f"score = {result.score}/10, passed = {result.passed}")
    print(f"--- Critic 輸出 ---\n{result.raw_output}\n--- 輸出結束 ---")

    # 期望：score < 8，且輸出含「論證邏輯」或「矛盾」
    if result.score >= 8:
        print("[WARN] 規則 5 可能未觸發（score 太高）— 請肉眼檢查上面輸出")
    elif "論證邏輯" in result.raw_output or "矛盾" in result.raw_output or "邏輯" in result.raw_output:
        print("[PASS] 規則 5 正確抓到邏輯/矛盾缺口")
    else:
        print("[WARN] score 低但未明確標出邏輯缺口，請檢查")
except Exception as e:
    print(f"[FAIL] critic_pipeline 例外：{type(e).__name__}: {e}")
    sys.exit(1)

# ── 測試 3：finance addendum（規則 6/7/8）─────────────────────────────────
print("\n=== 測試 3：finance addendum ===")
finance_report = """
台積電目標價 1200 元，使用 P/E 25x 估值（沒說 25x 是歷史均值還是同業均值）。
報告由 XX 投資銀行出具，給予買入評級（沒提是否與台積電有承銷關係）。
"""
from cowork.critic_pipeline import run_finance_critic_pipeline
try:
    result = run_finance_critic_pipeline(finance_report)
    print(f"score = {result.score}/10")
    has_finance = any(kw in result.raw_output for kw in ["估值框架", "利益衝突", "承銷"])
    if has_finance:
        print("[PASS] finance 規則 6/7 正確觸發")
    else:
        print("[WARN] 未明確標出 finance 規則，請檢查")
        print(f"輸出：{result.raw_output[:500]}")
except Exception as e:
    print(f"[FAIL] finance critic 例外：{type(e).__name__}: {e}")
    sys.exit(1)

print("\n=== 全部測試完成 ===")
