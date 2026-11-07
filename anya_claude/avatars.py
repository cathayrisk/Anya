# -*- coding: utf-8 -*-
"""聊天頭像（2026-11-07）：安妮亞依模式換表情，災防題一律 🌐。

- 一般模式＝開心大笑、深思模式＝「嘿嘿」得意臉、提問引導＝好奇的「嗯？」臉。
- 氣象署直答、以及所有災防相關的回合（走一般／深思也一樣）用 🌐：問的人可能正在擔心，不放笑臉
  （和狀態列的正經版台詞同一個原則，見頁面的 serious）。
- 認不得的模式（之後新增的、舊紀錄）用一般模式的臉，不會變成 Streamlit 預設的機器人圖示。
- 圖檔不在或打不開就退回改版前的 emoji：st.chat_message 拿到讀不到的圖會直接丟錯
  （"Failed to load the provided avatar value as an image."），那一則之後整頁都畫不出來。
  網頁上傳漏一張圖就會這樣，所以啟動時先檢查一次。
- 發問者（2026-11-07 加）：花生。安妮亞最愛吃花生，發問的人就是帶花生來的人；和頁面的 🥜 圖示呼應，
  所有同事共用一張。原創繪製（不用素材圖），要重畫或改色用 tools/draw-user-avatar.py。
  圖讀不到就回 None＝Streamlit 預設的發問者圖示。

圖：static/anya-avatar-{fast,deep,socratic,user}.png，128×128、256 色調色盤 PNG。不要存成一般 RGB PNG：
沒有透明通道的圖，Streamlit 每次畫都會重新壓成 JPEG（1.55 image_utils._validate_image_format_string）。
畫面上顯示成 32×32、圓角跟主題的 baseRadius、超出的邊裁掉（object-fit: cover）。
來源（使用者的 Desktop/Anya_icon；裁切框是原圖座標，縮到 128 後輕微銳利化）：
- 一般：aniyuki-anya-spy-x-family-26.gif 第 15 格，裁 (45, 0, 317, 272)，飽和 1.05
- 深思：244d9fbf-4e75-44c3-9af7-31ec057f551f.webp，裁 (55, 95, 395, 435)，底色 #FBE3D6，飽和 1.35、對比 1.15
- 提問引導：stickerfan_9029841_m_5ea7fc3bd9ce40cd26cde45db9665e91.webp 第 20 格，裁 (16, 0, 112, 96)，
  飽和 1.08、亮度 1.06

純 Python，不 import streamlit（anya_claude/ 的分層規則）。
"""
from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static"
FILES = {"fast": "anya-avatar-fast.png", "deep": "anya-avatar-deep.png", "socratic": "anya-avatar-socratic.png"}
EMOJI = {"fast": "⚡", "deep": "💬", "socratic": "🧭", "cwa": "🌐"}   # 改版前的頭像，也是缺圖時的備援
SERIOUS = "🌐"
DEFAULT_MODE = "fast"


def _usable(path: Path) -> bool:
    """檔案真的是讀得到的圖（上傳到一半壞掉的檔也擋掉）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def resolve(static_dir: Path = STATIC) -> dict[str, str]:
    """每個模式的頭像：圖能用就給路徑，不能用就給改版前的 emoji。"""
    table = dict(EMOJI)
    for mode, name in FILES.items():
        path = static_dir / name
        if path.is_file() and _usable(path):
            table[mode] = str(path)
    return table


AVATAR = resolve()   # 模組只載入一次：每個行程（Reboot 之後）檢查一次就好

USER_FILE = "anya-avatar-user.png"


def resolve_user(static_dir: Path = STATIC) -> str | None:
    """發問者的頭像：花生圖能用就給路徑，不能用就回 None（st.chat_message 會用 Streamlit 預設圖示）。"""
    path = static_dir / USER_FILE
    return str(path) if path.is_file() and _usable(path) else None


USER = resolve_user()


def for_mode(mode: str | None, *, serious: bool = False, table: dict[str, str] | None = None) -> str:
    """st.chat_message 的 avatar。serious＝災防相關的回合。"""
    if serious or mode == "cwa":
        return SERIOUS
    t = AVATAR if table is None else table
    return t.get(mode or "", t[DEFAULT_MODE])
