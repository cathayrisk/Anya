---
name: 安妮亞 (Anya)
description: 溫馨活潑的個人多功能助理工具集——以 SPY×FAMILY 安妮亞官方海報配色為本的「粉框＋紙頁」主題
colors:
  poster-pink: "#EF8E7F"
  cream: "#FFF7F0"
  ink: "#2E1F16"
  ink-heading: "#3B2A20"
  ink-h3: "#5C2E22"
  link-deep: "#5E241D"
  muted-canvas: "#4A362B"
  muted-surface: "#6B5A4E"
  coral-surface: "#C05A50"
  coral-body: "#A8433A"
  gold: "#C8A43A"
  brown: "#4A2F1A"
  primary-btn: "#AD4746"
  border-warm: "#D2AE9E"
  bg-card-end: "#FFF7F4"
  bg-panel: "#FFF1EC"
  stripe: "#FDF0ED"
  border-coral: "#F2D5CF"
  neutral-chip-bg: "#F1EDEA"
  code-bg: "#EDE8E6"
  rain-blue: "#1565C0"
  info-blue: "#33658E"
  warn-amber: "#8F5300"
  caution-gold: "#8A6D00"
  danger-red: "#B42318"
  success-green: "#054F31"
typography:
  body:
    fontFamily: "'SF Pro Rounded', 'PingFang TC', 'Microsoft JhengHei', 'Noto Sans CJK TC', 'WenQuanYi Micro Hei', sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.75
  heading:
    fontFamily: "'SF Pro Rounded', 'PingFang TC', 'Microsoft JhengHei', 'Noto Sans CJK TC', 'WenQuanYi Micro Hei', sans-serif"
    fontSize: "1.5rem"
    fontWeight: 700
    letterSpacing: "normal"
  mono:
    fontFamily: "'Cascadia Code', 'Consolas', 'Monaco', 'Menlo', 'DejaVu Sans Mono', monospace"
    fontSize: "0.92em"
rounded:
  sm: "8px"
  md: "12px"
  lg: "14px"
  pill: "99px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "14px"
  lg: "18px"
  xl: "20px"
components:
  card:
    backgroundColor: "{colors.bg-card-end}"
    textColor: "{colors.brown}"
    rounded: "{rounded.lg}"
    padding: "18px 20px 14px"
  chip-neutral:
    backgroundColor: "{colors.neutral-chip-bg}"
    textColor: "{colors.muted-surface}"
    rounded: "{rounded.pill}"
    padding: "4px 10px"
  chip-danger:
    backgroundColor: "#FDECEA"
    textColor: "{colors.danger-red}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  button-primary:
    backgroundColor: "{colors.primary-btn}"
    textColor: "{colors.cream}"
    rounded: "{rounded.sm}"
---

# Design System: 安妮亞 (Anya)

## 1. Overview

**Creative North Star: "海報畫框裡的信紙"（A Letter Framed in the Poster）**

這個主題的靈感來源是一張具體的圖：SPY×FAMILY 安妮亞官方海報——整面鮭魚粉上站著深褐制服、金邊袖口、白領子的安妮亞。但海報是看 3 秒的東西，app 是盯著讀半小時的東西，所以版式是「**粉框＋紙頁**」：**海報鮭魚粉（`#EF8E7F`）是畫布/畫框**，包圍整個內容區；**主內容區墊一張奶油紙面（`#FFFBF7`）**，所有長文（聊天回覆、報告、儀表板）都在紙上讀。海報感由畫框、側欄、chrome 承載；閱讀舒適由紙面保證——兩者不必互相犧牲。

金邊只出現在淺色或深色面上做裝飾（在粉底上金色幾乎隱形——實測對比 1.0，物理事實不是風格選擇）。系統明確拒絕：「一般 AI 生成的通用 SaaS 感」（漸層文字、無差別 pill 堆疊）、「過於嚴肅冷冰冰的企業後台」，以及自己的兩個舊版本——**近白粉畫布**（稀釋到沒有海報感）與**整面飽和粉直接放長文**（有海報感但閱讀疲勞）。

每一組文字/背景組合都經過 WCAG 公式實測：紙面上的珊瑚標題 5.78:1、內文 15.4:1；直接落在粉框上的少數文字另有深褐色票（6.68:1）。

**Key Characteristics:**
- 「粉框＋紙頁」：海報鮭魚粉當畫框，奶油紙面當閱讀層
- 長文永遠在紙上讀，海報感由 chrome（畫框/側欄/輸入區）承載
- 紙面上恢復完整的珊瑚暖色文字階層（標題/連結/清單符號）
- 安全警示（危險/警戒/注意）刻意跳出海報色，用紅/橘/藍語意色三重編碼
- 例外才浮出：平常沒事的狀態安靜，例外才搶眼

## 2. Colors

海報三色系統：粉畫布、褐墨、金邊，加上獨立的安全語意色軌。

**程式碼層的單一事實來源是 `utils/theme_tokens.py`**——這份文件的 frontmatter 是給人跟工具讀的規格說明，實際渲染顏色的程式碼（`utils/rich_styles.py`、`pages/11_氣象通知.py`）一律 `import utils.theme_tokens as tt` 取值，不重複硬編碼 hex。新增或修改顏色時，兩邊要一起改；只改這裡不改程式碼、或反過來，就是這個系統之前實際發生過的配色漂移事故。

### Primary
- **海報鮭魚粉 Poster Pink** (`#EF8E7F`): 畫框/畫布（`backgroundColor`）——包圍內容區、側欄之外、輸入區底，是海報識別的載體。
- **內容紙面 Paper** (`#FFFBF7`): 主內容區的閱讀層（`rich_styles` 將 block-container 墊成此色），所有長文在這上面。
- **制服深褐 Ink** (`#2E1F16`): 全站文字主色（`textColor`），紙面上 15.4:1、粉框上 6.68:1（兩種底都安全）。

### 紙面上的暖色文字階層（主要文字系統）
- **深珊瑚文字 Coral Text** (`#A8433A`): h1/h2 標題、連結、清單符號、定義詞（紙面上 5.78:1）。這是珊瑚識別色的「文字安全版」。
- **褐紅 H3/H4** (`#7A4030`): 次級標題（紙面上 7.83:1）。
- **暖紅 H5** (`#9A5040`)/**H6** (`#8A4A3C`): 更小的標題層級（5.67/6.53:1）。
- **暖褐灰 Muted** (`#6B5A4E`): 次要文字、刪除線、註腳（紙面上 6.38:1）。
- **珊瑚紅 Coral Decorative** (`#C05A50`): 純裝飾用途（blockquote 左框、checkbox accent）——當文字用一律換 `#A8433A`。

### 粉框上的文字色票（直接落在粉底上的少數文字，如未墊紙面的頁面）
- **連結/強調** (`#5E241D`): 粉底上 5.07:1（config 的 `linkColor`，兩種底都過）。
- 其餘文字盡量避免直接落在粉框上；必要時用 `#2E1F16`（6.68:1）。

### Secondary
- **金邊黃 Gold** (`#C8A43A`): 表格標頭文字（深褐底上）、blockquote 邊框——**只能放在深色或淺色面上**，粉底上對比 1.0 等於隱形。

### Neutral（表面層次）
- **奶油白 Cream** (`#FFF7F0`): 側欄、輸入框、widget 表面（`secondaryBackgroundColor`）——安妮亞的領子。
- **卡片漸層底** (`#FFFFFF → #FFF7F4`): 內容卡片，無描邊浮在粉底上。
- **現況面板底** (`#FFF1EC`): 卡片內的子面板，再深一階做層次。
- **斑馬紋** (`#FDF0ED`)、**淺面分隔線** (`#F2D5CF`)、**中性chip底** (`#F1EDEA`)、**程式碼底** (`#EDE8E6`): 只用於淺色面內部。
- **暖棕 widget 邊框** (`#D2AE9E`): Streamlit 原生 widget 的描邊（`borderColor`）。

### 互動色
- **按鈕主色 Primary Btn** (`#AD4746`): 深珊瑚紅按鈕，白字對比 5.58:1。

### 安全警示語意色（獨立軌，不與海報色混用）
- **警戒橘** (`#8F5300` on `#FFF3E0`)、**注意金棕** (`#8A6D00` on `#FEF7E6`)、**危險紅** (`#B42318` on `#FDECEA`)、**安全綠** (`#054F31` on `#ECFDF3`)
- **降雨藍** (`#1565C0` on `#E3F2FD`)、**資訊藍** (`#33658E`): 功能性藍色，雨水的通用隱喻。

### Named Rules
**The Frame-and-Page Rule.** 海報鮭魚粉是畫框，奶油紙面是內頁：長文閱讀只發生在紙上，海報感只由畫框與 chrome 承載。把飽和粉直接墊在長文底下、或把畫框淡化成近白，都是這條規則的違例。

**The Two-Surface Rule.** 每個顏色 token 都標注它屬於哪種底（紙面/粉框），跨底混用（例如把珊瑚文字放上粉框：1.84:1）就是對比度事故的開始。

**The Safety-Breaks-Brand Rule.** 影響人身安全的警示（颱風、特報、地震、健康指數）一律用紅/橘/藍語意色＋文字等級標籤＋圖示三重編碼，不套海報色，顏色永遠不是唯一判讀依據。

## 3. Typography

**Body/Display Font:** `'SF Pro Rounded', 'PingFang TC', 'Microsoft JhengHei', 'Noto Sans CJK TC', 'WenQuanYi Micro Hei', sans-serif`
**Mono Font:** `'Cascadia Code', 'Consolas', 'Monaco', 'Menlo', 'DejaVu Sans Mono', monospace`

**Character:** SF Pro Rounded 的圓潤字形配上海報粉畫布，可愛但字色永遠深到可讀。基準字級 16px。

### Hierarchy
- **H1** (700, 1.9em, 褐 hairline 底線): 畫布上用 `#3B2A20`。
- **H2** (700, 1.5em, 褐 hairline 底線): 同上。
- **H3/H4** (700, 1.28em/1.08em): `#5C2E22`。
- **H5/H6** (600, 0.95em/0.87em): `#4A362B`。
- **Body** (400, 1rem, line-height 1.75): `#2E1F16`（theme textColor）。
- **Caption/Label** (0.7–0.85rem): 表面上用 `#6B5A4E`，畫布上用 `#4A362B`。

### Named Rules
**The No Tiny Text Rule.** 家人裡有非技術背景的使用者，資訊性文字不低於 0.7rem，白色文字在粉底上即使是大字也禁用於功能性內容（實測 2.37:1 連大字標準都不過——海報的白色標題字是純裝飾，我們的內容不是）。

## 4. Elevation

系統不用陰影；深度有三層表達：**粉框→紙面用明度對比**（奶油紙頁浮在飽和粉框上，18px 圓角，無描邊無陰影），**紙面→卡片用淡珊瑚描邊**（`#F2D5CF` 1px，因為兩層都是淺色、需要一條線分離），**卡片內部用暖色調深淺分層**（白 → `#FFF7F4` → `#FFF1EC`）。

### Named Rules
**The Page-on-Poster Rule.** 紙頁浮在粉框上靠明度對比，不加邊框陰影；紙面上的卡片彼此都是淺色，靠淡珊瑚描邊分離。哪一層用哪種分離手段是固定的，不要混用。

### 已知限制：沒有深色模式
`.streamlit/config.toml` 的 `[theme]` 只能定義一組固定配色，Streamlit 不支援像網頁 CSS 的 `prefers-color-scheme` 那樣自動跟隨系統深色模式切換到不同色票；使用者若從 Streamlit 選單手動切換，介面可能不會有對應的深色版本呼應。這是稽核時發現、經確認後刻意不處理的已知限制，不是遺漏——如果之後真的需要深色模式，那是一次獨立的設計工作——深色模式不是直接反轉亮色的顏色，深度表達方式（明度分層 vs 陰影）、強調色飽和度都要重新設計，不是這次順手加得完的。

## 5. Components

### Cards
- **Corner Style:** 14px 圓角，內部子面板 12px（紙頁本身 18px——圓角半徑隨層級遞減）。
- **Background:** `linear-gradient(180deg, #FFFFFF 0%, #FFF7F4 100%)`。
- **Border:** `#F2D5CF` 1px——卡片坐在紙面上，兩層都是淺色，靠描邊分離（見 The Page-on-Poster Rule）。
- **Internal Padding:** 18–20px 外層、12–14px 內部。

### Chips / Badges
- 中性（沒事）：`#F1EDEA` 底/`#6B5A4E` 字，低調。
- 降雨功能色：藍系生效中、橘系即將發生。
- 安全警示：`fill-*`（實心＝現在生效）與 `outline-*`（外框＝預測將達），同色系深淺表達急迫度，位置固定不搬家。
- 全部 99px pill 圓角。

### 警示橫幅 (Alert Banner)
- 左圖示＋明文等級 chip（白字實色底：危險紅 `#B42318`/警戒橘 `#B54708`/注意藍 `#175CD3`）＋標題＋在地標記。自帶淺色底（`#FFF1F0`/`#FFFAEB`/`#EFF8FF`），在粉畫布上是明確的「事件發生了」訊號。

### 溫度帶 (Temperature Band)
- 簽名元件：冷藍 `rgb(110,159,197)` → 金 → 暖橘 → 珊瑚紅 `rgb(192,90,80)` 依溫度插值的漸層長條，放在卡片淺底上。

### Streamlit 原生 Widget
- 靠 `config.toml` 主題統一：奶油面（`secondaryBackgroundColor`）、暖棕描邊（`#D2AE9E`）、深珊瑚按鈕（`#AD4746`）。

### 全站 Markdown 主題（`utils/rich_styles.py`）
- 直接渲染在粉畫布上，所有顏色屬於「畫布上的文字色票」組；表格/blockquote/程式碼區塊自帶淺色底，內部才能用表面色票。表格的 td 自帶 `#FFFDFB` 底色，不讓粉底透進奇數列。

## 6. Do's and Don'ts

### Do:
- **Do** 加任何新文字前先問：它落在粉畫布上還是奶油表面上？然後從對應那組色票選色。
- **Do** 讓淺色表面靠明度對比浮在粉底上，不加描邊。
- **Do** 顏色狀態指示永遠搭配文字說明（「危險」「警戒」），色彩是第三重輔助。
- **Do** 同一資訊概念固定位置，狀態變化只換樣式深淺。
- **Do** 新顏色上線前用 WCAG 公式驗證（本文件所有數字都是實測值）。

### Don't:
- **Don't** 把畫框調回近白粉（`#FFF6F7` 當 backgroundColor）——海報的粉紅是飽和的；也 **don't** 把長文直接放上飽和粉底——那是閱讀疲勞。兩個都是走過的彎路。
- **Don't** 在粉框上直接使用珊瑚色（1.84:1）、金色（1.0:1）、白色功能文字（2.37:1）、或任何「紙面色票」的灰。
- **Don't** 用漸層文字（`background-clip:text`）。
- **Don't** 把同色系 pill 無差別堆疊成一排。
- **Don't** 把安全警示的紅/橘/藍跟海報色混用——兩套色彩系統刻意分離。
- **Don't** 引入冷色系灰階（`#172033`/`#475467`/`#667085` 這類 slate 色號），連邊框都不要。
- **Don't** 只靠顏色區分狀態或嚴重度。
- **Don't** 用側邊條紋（`border-left`/`border-right` > 1px）當作卡片、callout、blockquote 的彩色強調——這個系統的 blockquote 曾經這樣做過，已改成全邊框；不要再犯。強調用全邊框、底色暈染、或前導圖示，不用側邊條紋。
- **Don't** 顏色只改文件（本檔）不改 `utils/theme_tokens.py`，或只改程式碼不改本檔——兩邊要保持同步。
