# utils/theme_tokens.py
# -*- coding: utf-8 -*-
"""
Anya 設計系統的顏色 token 單一事實來源，對應 DESIGN.md 的 frontmatter colors 區塊。

背景：氣象通知頁的配色曾經在迭代過程中不知不覺漂移，跟全站主題脫節，因為顏色
是分別硬編碼在 utils/rich_styles.py 跟 pages/11_氣象通知.py 兩處、各自維護。
之後任何檔案要用到品牌色，一律 import 這裡的常數，不要重新寫 hex 字面值——
改色只改這一個檔案，其他地方自動跟著更新，不會再重演那次事故。

命名對齊 DESIGN.md 的 token 名稱（kebab-case → SCREAMING_SNAKE_CASE）。
"""

from __future__ import annotations

# ── 畫框／紙頁（見 .streamlit/config.toml 的 backgroundColor/secondaryBackgroundColor）──
POSTER_PINK = "#EF8E7F"   # 海報鮭魚粉（畫框，僅 config.toml 使用）
CREAM = "#FFF7F0"         # 奶油白表面（側欄/輸入框，僅 config.toml 使用）
PAPER = "#FFFBF7"         # 內容紙面（主內容區閱讀層）

# ── 紙面上的暖色文字階層 ──
INK = "#2E1F16"                  # 全站文字主色
CORAL_TEXT = "#A8433A"           # 珊瑚色系文字安全版（h1/h2/連結/清單符號，紙面上 5.78:1）
CORAL_DECORATIVE = "#C05A50"     # 純裝飾用珊瑚（邊框/checkbox accent）——禁止當文字用在粉框上
H3_BROWN = "#7A4030"             # h3/h4
H5_BROWN = "#9A5040"             # h5
H6_BROWN = "#8A4A3C"             # h6
MUTED = "#6B5A4E"                # 次要文字/刪除線/註腳（紙面上 6.38:1）
BROWN = "#4A2F1A"                # 制服深褐（表格標頭底色等）
GOLD = "#C8A43A"                 # 金邊黃——僅深色/淺色面裝飾，粉框上幾乎隱形

# ── 淺色面（卡片／表格／blockquote／chip 內部）──
LIGHT = "#FFF5F2"          # 淡珊瑚內襯底
BG_CARD_END = "#FFF7F4"    # 卡片漸層終點色
BG_PANEL = "#FFF1EC"       # 卡片內子面板（比卡片底再深一階）
STRIPE = "#FDF0ED"         # 斑馬紋偶數列
BORDER = "#F2D5CF"         # 淺色面分隔線／卡片描邊
CODE_BG = "#EDE8E6"        # 行內程式碼底色
NEUTRAL_CHIP_BG = "#F1EDEA"  # 「沒事」狀態 chip 底色
HEALTH_ACTIVE_BG = "#FBF4F3"  # 健康警示現在生效中的列底色（淡珊瑚，比卡片底再深一階）

# ── 互動色 ──
PRIMARY_BTN = "#AD4746"    # 按鈕主色（僅 config.toml 使用）
BORDER_WARM = "#D2AE9E"    # Streamlit widget 描邊（僅 config.toml 使用）

# ── 功能語意色（降雨等，非安全警示）──
RAIN_BLUE = "#1565C0"
RAIN_BLUE_BG = "#E3F2FD"
INFO_BLUE = "#33658E"      # 降雨機率百分比

# ── 安全警示語意色（獨立軌，刻意不與品牌珊瑚/金色混用，見 DESIGN.md Safety-Breaks-Brand Rule）──
WARN_AMBER = "#8F5300"
WARN_AMBER_BG = "#FFF3E0"
CAUTION_GOLD = "#8A6D00"
CAUTION_GOLD_BG = "#FEF7E6"
DANGER_RED = "#B42318"
DANGER_RED_BG = "#FDECEA"
SUCCESS_GREEN = "#054F31"
SUCCESS_GREEN_BG = "#ECFDF3"

# 警示橫幅（al-danger/al-watch/al-notice）用的較淡背景+邊框，跟上面的 chip 底色是同一色相
# 但明度不同層級，兩組分開命名以保留各自語意
ALERT_DANGER_BG = "#FFF1F0"
ALERT_DANGER_BORDER = "#F1AEA7"
ALERT_WATCH_BG = "#FFFAEB"
ALERT_WATCH_BORDER = "#F0D28E"
ALERT_WATCH_CHIP = "#B54708"
ALERT_NOTICE_BG = "#EFF8FF"
ALERT_NOTICE_BORDER = "#B5D3F5"
ALERT_NOTICE_CHIP = "#175CD3"
ALERT_STRIP_HIT_TEXT = "#7A1710"
ALERT_SUCCESS_BORDER = "#A6E4C0"

# 健康/天氣警示 outline badge（稍後才會達標）用的邊框色，比 fill 版本淡
OUTLINE_DANGER_BORDER = "#F1AEA7"
OUTLINE_WARN_BORDER = "#F0D28E"
OUTLINE_CAUTION_BORDER = "#F2E2A0"
OUTLINE_SAFE_BORDER = "#E8DDD6"

# ── 溫度視覺化長條的漸層端點（冷藍 → 金 → 暖橘 → 珊瑚紅）──
TEMP_GRADIENT_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (12, (110, 159, 197)),
    (22, (200, 164, 58)),
    (30, (224, 138, 99)),
    (36, (192, 90, 80)),
]
