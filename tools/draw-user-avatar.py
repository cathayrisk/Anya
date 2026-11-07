# -*- coding: utf-8 -*-
"""重畫發問者頭像（花生，2026-11-07）：static/anya-avatar-user.png。

原創繪製，不讀任何圖檔。512px 畫布 → 縮到 128（LANCZOS）→ 輕微銳利化 → 256 色調色盤 PNG
（和安妮亞頭像同一套流程；純 RGB PNG 會被 Streamlit 每次重壓成 JPEG，見 anya_claude/avatars.py）。
畫面上顯示 32px、圓角 10px，四角會被裁掉，所以花生收在中間。
顏色取自 Anya 主題：底色＝secondaryBackgroundColor #FFDFE0，花生用褐色系。

用法：python tools/draw-user-avatar.py [輸出路徑]   （不給路徑就覆蓋 static/anya-avatar-user.png）
"""
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

S = 512
BG = "#FFDFE0"
OUTLINE = "#7A4A2A"
SHELL = "#DDAA6A"
DIMPLE = (176, 122, 66, 255)
SHINE = (248, 222, 172, 230)


def _circle(d: ImageDraw.ImageDraw, cx, cy, r, fill):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)


def draw() -> Image.Image:
    bg = Image.new("RGB", (S, S), BG)
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    lobes = [(256, 178, 92), (256, 326, 104)]          # 上小下大，兩圓重疊出腰身
    w = 16
    for cx, cy, r in lobes:                              # 先畫外框（半徑多 w）再填色 → 兩圓聯集的外框
        _circle(d, cx, cy, r + w, OUTLINE)
    for cx, cy, r in lobes:
        _circle(d, cx, cy, r, SHELL)
    mask = Image.new("L", (S, S), 0)                     # 殼上的網紋（小凹點），只畫在殼內
    md = ImageDraw.Draw(mask)
    for cx, cy, r in lobes:
        _circle(md, cx, cy, r - 18, 255)
    dots = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots)
    for row, y in enumerate(range(96, 440, 38)):
        off = 19 if row % 2 else 0
        for x in range(150 + off, 380, 38):
            dd.ellipse((x - 9, y - 6, x + 9, y + 6), fill=DIMPLE)
    dots.putalpha(ImageChops.multiply(dots.getchannel("A"), mask))
    layer = Image.alpha_composite(layer, dots)
    hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))         # 亮面
    hd = ImageDraw.Draw(hl)
    hd.ellipse((196, 118, 262, 160), fill=SHINE)
    hd.ellipse((190, 262, 262, 306), fill=SHINE)
    layer = Image.alpha_composite(layer, hl)
    layer = layer.rotate(-32, resample=Image.BICUBIC, center=(256, 252))
    bg.paste(layer, (0, 6), layer)
    return bg


def save(im: Image.Image, path: Path) -> None:
    im = im.convert("RGB").resize((128, 128), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=2))
    q = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
    q.save(path, optimize=True)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "static" / "anya-avatar-user.png"
    save(draw(), out)
    print(out)
