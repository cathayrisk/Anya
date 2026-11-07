# -*- coding: utf-8 -*-
"""聊天頭像（anya_claude/avatars.py，2026-11-07）：依模式換安妮亞的表情、災防題一律 🌐、缺圖退回 emoji；發問者用花生。"""
import ast
import shutil
from pathlib import Path

from anya_claude import avatars as A

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "7_Anya_Claude.py"


def test_shipped_avatars_are_small_square_pngs_streamlit_passes_through():
    from PIL import Image
    for name in [*A.FILES.values(), A.USER_FILE]:
        path = A.STATIC / name
        assert path.is_file(), f"static/{name} 不見了（網頁上傳漏傳？）"
        with Image.open(path) as im:
            assert im.format == "PNG" and im.size == (128, 128)
            # 沒有透明通道的 RGB 圖，Streamlit 每次畫都會重新壓成 JPEG；調色盤／RGBA 才會原檔照送
            assert im.mode in {"P", "RGBA"}, im.mode
        assert path.stat().st_size < 40_000


def test_resolve_uses_the_images():
    table = A.resolve()
    for mode, name in A.FILES.items():
        assert table[mode].endswith(name) and Path(table[mode]).is_file()
    assert table["cwa"] == "🌐"
    assert A.resolve_user() == str(A.STATIC / A.USER_FILE) == A.USER


def test_user_avatar_missing_or_broken_falls_back_to_streamlit_default(tmp_path):
    assert A.resolve_user(tmp_path) is None                         # 花生圖沒傳
    (tmp_path / A.USER_FILE).write_bytes(b"not a png")              # 上傳到一半壞掉
    assert A.resolve_user(tmp_path) is None
    shutil.copy(A.STATIC / A.USER_FILE, tmp_path / A.USER_FILE)
    assert A.resolve_user(tmp_path) == str(tmp_path / A.USER_FILE)


def test_missing_or_broken_image_falls_back_to_emoji(tmp_path):
    shutil.copy(A.STATIC / A.FILES["fast"], tmp_path / A.FILES["fast"])
    (tmp_path / A.FILES["deep"]).write_bytes(b"not a png")   # 上傳到一半壞掉
    table = A.resolve(tmp_path)                               # 提問引導那張整個沒傳
    assert table["fast"] == str(tmp_path / A.FILES["fast"])
    assert table["deep"] == "💬" and table["socratic"] == "🧭" and table["cwa"] == "🌐"


def test_for_mode_rules():
    t = {"fast": "F", "deep": "D", "socratic": "S", "cwa": "🌐"}
    assert A.for_mode("fast", table=t) == "F"
    assert A.for_mode("deep", table=t) == "D"
    assert A.for_mode("socratic", table=t) == "S"
    assert A.for_mode("cwa", table=t) == "🌐"
    for mode in ("fast", "deep", "socratic", None):
        assert A.for_mode(mode, serious=True, table=t) == "🌐"   # 災防題不放笑臉
    # 認不得的模式（舊紀錄、之後新增的）：一般模式的臉，不是 Streamlit 預設的機器人圖示
    assert A.for_mode(None, table=t) == "F"
    assert A.for_mode("research", table=t) == "F"


def _page_tree() -> ast.Module:
    return ast.parse(PAGE.read_text(encoding="utf-8"))


def test_every_assistant_bubble_gets_its_avatar_from_for_mode():
    calls = [n for n in ast.walk(_page_tree()) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "chat_message"]
    assistant = [c for c in calls
                 if not (c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "user")]
    assert len(assistant) >= 4   # 新回合、斷線接續、歷史重畫、放棄接續的那一則
    for c in assistant:
        av = next((k.value for k in c.keywords if k.arg == "avatar"), None)
        assert av is not None and "A.for_mode(" in ast.unparse(av), ast.unparse(c)


def test_every_user_bubble_gets_the_peanut():
    """發問者泡泡（剛送出的、重畫歷史的）都用花生；歷史迴圈是 role 變數，助理用 for_mode、發問者用花生。"""
    calls = [n for n in ast.walk(_page_tree()) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "chat_message"]
    user = [c for c in calls if c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "user"]
    either = [c for c in calls if c.args and isinstance(c.args[0], ast.Name)]
    assert user and either
    for c in user + either:
        av = next((k.value for k in c.keywords if k.arg == "avatar"), None)
        assert av is not None and "USER_AVATAR" in ast.unparse(av), ast.unparse(c)


def test_page_survives_an_old_avatars_module():
    """只上傳了頁面、還沒 Reboot 時，記憶體裡是舊的 avatars（沒有 USER）：頁面要用 getattr 取，不能直接 A.USER。"""
    tree = _page_tree()
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "USER_AVATAR" for t in n.targets)]
    assert len(assigns) == 1
    assert ast.unparse(assigns[0].value) == "getattr(A, 'USER', None)"
    assert not any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "A"
                   and n.attr == "USER" for n in ast.walk(tree))


def test_saved_answers_remember_if_it_was_a_hazard_turn():
    """重畫歷史時頭像要和當下一致：存進歷史的助理回答都帶 serious（氣象署直答本來就是 🌐，不用帶）。"""
    checked = 0
    for n in ast.walk(_page_tree()):
        if not isinstance(n, ast.Dict):
            continue
        keys = {k.value: v for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant)}
        role, mode = keys.get("role"), keys.get("mode")
        if not (isinstance(role, ast.Constant) and role.value == "assistant"):
            continue
        if isinstance(mode, ast.Constant) and mode.value == "cwa":
            continue
        assert "serious" in keys, ast.unparse(n)
        checked += 1
    assert checked >= 3   # 答完、按停止、放棄接續
