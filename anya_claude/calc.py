# -*- coding: utf-8 -*-
"""精確計算（calculate 工具的核心）。純 Python，無 streamlit 依賴——Haiku 節點與 deep agent 的工具都在背景執行緒呼叫。

為什麼要有：2026-10-24 實測 Haiku 心算「每月存 1 萬、年報酬 5%、20 年」答約 420 萬，照它自己寫的
月複利公式應為 4,110,336（差 2%）。金融同事最常問的就是這種試算，模型不該心算。

安全：不用 eval。先 ast.parse，只放行白名單裡的節點（數字、四則、次方、括號、變數、清單）與函式，
其他一律回錯誤；次方與結果大小有上限，算不完的式子也回錯誤，不會卡住或吃光記憶體。

財務函式和 Excel 同名、同參數順序、同正負號慣例（付出為負、收到為正），模型最熟這一套：
  fv(rate, nper, pmt, pv=0, when=0)   pv(rate, nper, pmt, fv=0, when=0)
  pmt(rate, nper, pv, fv=0, when=0)   nper(rate, pmt, pv, fv=0, when=0)
  npv(rate, [CF0, CF1, …])：CF0 在第 0 期不折現（和 Excel 的 NPV 不同，Excel 從第 1 期起折現）
  irr([CF0, CF1, …])
when=1 表示期初（先存再計息）。
"""
from __future__ import annotations

import ast
import math
from typing import Any

MAX_LINES = 20
MAX_CHARS = 300
MAX_EXPONENT = 10_000
MAX_ABS = 1e300


class CalcError(ValueError):
    pass


# ───────── 財務函式（Excel 慣例）─────────
def fv(rate, nper, pmt, pv=0.0, when=0):
    if rate == 0:
        return -(pv + pmt * nper)
    g = (1 + rate) ** nper
    return -(pv * g + pmt * (1 + rate * when) / rate * (g - 1))


def pv(rate, nper, pmt, fv=0.0, when=0):   # noqa: A002 — 和 Excel 同名
    if rate == 0:
        return -(fv + pmt * nper)
    g = (1 + rate) ** nper
    return -(fv + pmt * (1 + rate * when) / rate * (g - 1)) / g


def pmt(rate, nper, pv, fv=0.0, when=0):   # noqa: A002
    if nper == 0:
        raise CalcError("nper（期數）不能是 0")
    if rate == 0:
        return -(fv + pv) / nper
    g = (1 + rate) ** nper
    return -(fv + pv * g) * rate / ((1 + rate * when) * (g - 1))


def nper(rate, pmt, pv, fv=0.0, when=0):   # noqa: A002
    if rate == 0:
        if pmt == 0:
            raise CalcError("利率和每期金額都是 0，期數算不出來")
        return -(fv + pv) / pmt
    a = pmt * (1 + rate * when)
    num, den = a - fv * rate, a + pv * rate
    if den == 0 or num / den <= 0:
        raise CalcError("這組金額永遠達不到目標，期數算不出來（檢查正負號：付出為負、收到為正）")
    return math.log(num / den) / math.log(1 + rate)


def npv(rate, values):
    vals = _as_list(values, "npv")
    return sum(v / (1 + rate) ** t for t, v in enumerate(vals))


def irr(values):
    vals = _as_list(values, "irr")
    if not (any(v > 0 for v in vals) and any(v < 0 for v in vals)):
        raise CalcError("irr 的現金流要同時有正有負")
    f = lambda r: sum(v / (1 + r) ** t for t, v in enumerate(vals))  # noqa: E731
    lo, hi = -0.9999, 1.0
    while f(lo) * f(hi) > 0 and hi < 1e6:
        hi *= 2
    if f(lo) * f(hi) > 0:
        raise CalcError("找不到使淨現值為 0 的報酬率")
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _as_list(values, name):
    if not isinstance(values, list) or not values:
        raise CalcError(f"{name} 的現金流要用清單，例如 [-100, 30, 40, 50]")
    return values


def _round(x, n=0):
    return round(x, int(n))


def _power(a, b):
    """次方（** 和 pow 共用）：指數、結果大小都有上限，避免 10**10**10 這種把伺服器卡死的式子。"""
    if abs(b) > MAX_EXPONENT:
        raise CalcError("次方太大")
    if a < 0 and not float(b).is_integer():
        raise CalcError("負數不能開非整數次方")
    if isinstance(a, int) and isinstance(b, int) and b > 0 and abs(a) > 1 and b * math.log10(abs(a)) > 300:
        raise CalcError("結果太大")
    return a ** b


FUNCS = {
    "abs": abs, "round": _round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "ceil": math.ceil, "floor": math.floor, "pow": _power,
    "fv": fv, "pv": pv, "pmt": pmt, "nper": nper, "npv": npv, "irr": irr,
}
CONSTS = {"pi": math.pi, "e": math.e}

_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b}


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise CalcError("只能算數字")
    if (isinstance(x, float) and math.isnan(x)) or abs(x) > MAX_ABS:
        raise CalcError("結果太大或不是數字")
    return x


def _eval(node, env: dict):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return _num(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in CONSTS:
            return CONSTS[node.id]
        raise CalcError(f"不認得「{node.id}」（變數要先在前面一行定義，例如 r = 0.05/12）")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _num(_eval(node.operand, env))
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = _num(_eval(node.left, env)), _num(_eval(node.right, env))
        if isinstance(node.op, ast.Pow):
            return _num(_power(a, b))
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise CalcError("只支援 + - * / // % **")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
            raise CalcError("除以 0")
        return _num(op(a, b))
    if isinstance(node, ast.List):
        return [_num(_eval(x, env)) for x in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS:
        args = [_eval(a, env) for a in node.args]
        kwargs = {k.arg: _eval(k.value, env) for k in node.keywords if k.arg}
        if len(kwargs) != len(node.keywords):
            raise CalcError("不支援 **參數")
        out = FUNCS[node.func.id](*args, **kwargs)
        return out if isinstance(out, list) else _num(out)
    if isinstance(node, ast.Tuple):
        raise CalcError("數字不要加千分位逗號（1,000 寫成 1000）；多筆現金流用 [ ]")
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", "?")
        raise CalcError(f"不支援函式「{name}」；可用：{', '.join(FUNCS)}")
    raise CalcError("只支援數字、四則運算、次方、括號、變數和清單")


def fmt(value) -> str:
    """給人看的寫法：千分位；整數不帶小數；一般數字兩位小數；很小的數字保留有效位數。"""
    if isinstance(value, list):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    if isinstance(value, int) or float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.6g}"


def _parse_line(line: str):
    """「名稱 = 算式」或純算式。回 (名稱或 None, 算式的 AST)。"""
    try:
        tree = ast.parse(line, mode="exec")
    except SyntaxError:
        raise CalcError("算式寫法有誤（百分比請用小數，例如 5% 寫 0.05；次方用 **）") from None
    if len(tree.body) != 1:
        raise CalcError("一行只放一個算式")
    stmt = tree.body[0]
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        name = stmt.targets[0].id
        if name in FUNCS or name in CONSTS:
            raise CalcError(f"「{name}」是保留名稱，換個變數名")
        return name, ast.Expression(stmt.value)
    if isinstance(stmt, ast.Expr):
        return None, ast.Expression(stmt.value)
    raise CalcError("只支援「名稱 = 算式」或純算式")


def evaluate(lines: list[str]) -> list[dict[str, Any]]:
    """逐行計算；前面定義的變數後面可以用。每行回 {expr, name?, value, text} 或 {expr, error}。"""
    if isinstance(lines, str):
        lines = [lines]
    env: dict[str, Any] = {}
    out: list[dict[str, Any]] = []
    for raw in list(lines or [])[:MAX_LINES]:
        line = str(raw).strip()
        rec: dict[str, Any] = {"expr": line[:MAX_CHARS]}
        try:
            if not line:
                raise CalcError("空白")
            if len(line) > MAX_CHARS:
                raise CalcError(f"一行最多 {MAX_CHARS} 字")
            name, tree = _parse_line(line)
            value = _eval(tree, env)
            if name:
                env[name] = value
                rec["name"] = name
            rec["value"] = value
            rec["text"] = fmt(value)
        except CalcError as e:
            rec["error"] = str(e)
        except (OverflowError, ZeroDivisionError, ValueError, TypeError, RecursionError) as e:
            rec["error"] = f"算不出來：{type(e).__name__}"
        out.append(rec)
    if len(lines or []) > MAX_LINES:
        out.append({"expr": "…", "error": f"一次最多 {MAX_LINES} 行"})
    return out
