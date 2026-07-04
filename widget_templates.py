# -*- coding: utf-8 -*-
"""
widget_templates.py — Anya 互動 HTML widget 模板庫

純 Python 常數模組（零依賴、不 import streamlit）。
由 pages/Anya_Gemma.py 匯入：LLM 透過 load_skill 讀取模板骨架，
替換資料區後呼叫 create_widget(title, height, html)，
以 st.components.v1.html(html, height=..., scrolling=True) 渲染。

設計哲學：單檔、零依賴、開箱即互動（參考 anthropics/html-effectiveness）。
"""
import re

# 使用者訊息若命中此 pattern，提示模型考慮使用 widget
WIDGET_HINT_RE = re.compile(
    r"計算機|互動(?:表格|元件|比較)|比較矩陣|抽認卡|翻卡|小測驗|做成.{0,4}(?:工具|元件)|widget",
    re.IGNORECASE,
)

# 注入 system prompt 的規則片段
WIDGET_RULES = """【互動 widget 規則】
何時使用：數字試算或參數探索 → widget_calculator；多對象多維度比較 → widget_comparison_matrix；研究來源探索 → widget_source_browser；從文件重點學習 → widget_flashcards。
何時不用：答案用敘述文字就能講清楚時，不要做 widget；每回合最多 1 個 widget。
流程：先用 load_skill 載入對應模板 → 只替換「資料區」內的 DATA（與範例同結構的真實資料），不改結構、CSS 與 JS 邏輯 → 呼叫 create_widget(title, height, html)。
硬規則：HTML 必須完全自包含；禁止外部資源（CDN、字體、圖片 URL）；禁止 fetch/XHR；height 依內容估 200-800。"""


WIDGET_TEMPLATES: dict[str, dict] = {

    # ------------------------------------------------------------------
    "widget_comparison_matrix": {
        "description": "互動比較矩陣：2-4 個對象 × 4-8 個維度，維度可勾選顯示/隱藏，點格子標記優勝方。",
        "content": r"""用途：多個對象跨維度的互動比較表（凍結表頭、維度篩選、優勝方標記）。
可改：僅「資料區」內的 const DATA — title、items（2-4 個對象）、criteria（4-8 個維度；values 依 items 順序；winner 為優勝方 index 或 null）。
不可動：資料區以外的 HTML 結構、CSS 與 JS 邏輯。
height 建議：依維度數估 380-560。

<style>
#anya-cmx{--brand:#AD4746;--bg:#FFF6F7;--surface:#FFDFE0;--ink:#4B3832;--radius:10px;
  font-family:"SF Pro Rounded",-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif;
  background:var(--bg);color:var(--ink);padding:16px;border-radius:var(--radius);}
#anya-cmx h3{margin:0 0 4px;font-size:17px;}
#anya-cmx .cmx-count{font-size:12px;opacity:.75;margin-bottom:10px;}
#anya-cmx .cmx-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;}
#anya-cmx .cmx-chip{background:var(--surface);border:1px solid transparent;border-radius:999px;
  padding:3px 10px;font-size:12px;cursor:pointer;transition:all .2s ease;user-select:none;}
#anya-cmx .cmx-chip.off{background:transparent;border-color:var(--surface);opacity:.55;}
#anya-cmx .cmx-scroll{max-height:360px;overflow:auto;border-radius:var(--radius);
  border:1px solid var(--surface);background:#fff;}
#anya-cmx table{border-collapse:collapse;width:100%;font-size:13px;}
#anya-cmx th,#anya-cmx td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--surface);}
#anya-cmx thead th{position:sticky;top:0;background:var(--surface);z-index:1;}
#anya-cmx td.val{cursor:pointer;transition:background .2s ease,color .2s ease;}
#anya-cmx td.val.win{background:var(--brand);color:#fff;font-weight:600;}
#anya-cmx .cmx-hint{font-size:11px;opacity:.6;margin-top:8px;}
</style>
<div id="anya-cmx">
  <h3 id="cmx-title"></h3>
  <div class="cmx-count" id="cmx-count"></div>
  <div class="cmx-chips" id="cmx-chips"></div>
  <div class="cmx-scroll"><table>
    <thead id="cmx-head"></thead><tbody id="cmx-body"></tbody>
  </table></div>
  <div class="cmx-hint">點膠囊可顯示／隱藏維度；點任一格可標記該列優勝方。</div>
</div>
<!-- ══ 資料區：以下由模型替換 ══ -->
<script>
const DATA = {
  title: "2026 輕薄筆電比較",
  items: ["MacBook Air 13", "ASUS Zenbook 14", "Acer Swift Go 14"],   // 2-4 個比較對象
  // criteria：4-8 個維度。name=維度名；values=各對象的值（依 items 順序）；winner=優勝方 index（無則 null）
  criteria: [
    { name: "價格",   values: ["NT$ 35,900", "NT$ 31,900", "NT$ 27,900"], winner: 2 },
    { name: "重量",   values: ["1.24 kg", "1.19 kg", "1.32 kg"],          winner: 1 },
    { name: "續航",   values: ["約 18 小時", "約 12 小時", "約 10 小時"],  winner: 0 },
    { name: "螢幕",   values: ["13.6 吋 IPS", "14 吋 OLED", "14 吋 OLED"], winner: null },
    { name: "記憶體", values: ["16GB 統一記憶體", "16GB", "16GB"],        winner: null },
    { name: "保固",   values: ["1 年", "2 年", "2 年"],                   winner: null }
  ]
};
</script>
<!-- ══ 資料區結束 ══ -->
<script>
(function () {
  const on = DATA.criteria.map(() => true);
  document.getElementById("cmx-title").textContent = DATA.title;
  const head = document.getElementById("cmx-head");
  head.innerHTML = "<tr><th>維度</th>" +
    DATA.items.map(function (i) { return "<th></th>"; }).join("") + "</tr>";
  head.querySelectorAll("th").forEach(function (th, k) {
    if (k > 0) th.textContent = DATA.items[k - 1];
  });
  const chips = document.getElementById("cmx-chips");
  DATA.criteria.forEach(function (c, ci) {
    const chip = document.createElement("span");
    chip.className = "cmx-chip";
    chip.textContent = "✓ " + c.name;
    chip.addEventListener("click", function () {
      on[ci] = !on[ci];
      chip.classList.toggle("off", !on[ci]);
      chip.textContent = (on[ci] ? "✓ " : "– ") + c.name;
      render();
    });
    chips.appendChild(chip);
  });
  function render() {
    document.getElementById("cmx-count").textContent =
      "目前勾選 " + on.filter(Boolean).length + "/" + DATA.criteria.length + " 維度";
    const body = document.getElementById("cmx-body");
    body.innerHTML = "";
    DATA.criteria.forEach(function (c, ci) {
      if (!on[ci]) return;
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = c.name;
      name.style.fontWeight = "600";
      tr.appendChild(name);
      c.values.forEach(function (v, vi) {
        const td = document.createElement("td");
        td.className = "val" + (c.winner === vi ? " win" : "");
        td.textContent = v;
        td.addEventListener("click", function () {
          c.winner = (c.winner === vi ? null : vi);
          render();
        });
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }
  render();
})();
</script>""",
    },

    # ------------------------------------------------------------------
    "widget_calculator": {
        "description": "參數計算機：2-4 個滑桿+數字框雙向同步，即時重算並顯示主結果與帶入值。",
        "content": r"""用途：數字試算／參數探索（滑桿與數字框雙向同步，即時重算）。
可改：僅「資料區」內的 const DATA — title、formula 說明、inputs（2-4 個參數：id/label/min/max/step/default/unit）、compute 函式（回傳 {main, detail}，模型改這裡）。
不可動：資料區以外的 HTML 結構、CSS 與 JS 邏輯。
height 建議：依參數數估 320-480。

<style>
#anya-calc{--brand:#AD4746;--bg:#FFF6F7;--surface:#FFDFE0;--ink:#4B3832;--radius:10px;
  font-family:"SF Pro Rounded",-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif;
  background:var(--bg);color:var(--ink);padding:16px;border-radius:var(--radius);}
#anya-calc h3{margin:0 0 12px;font-size:17px;}
#anya-calc .calc-row{margin-bottom:12px;}
#anya-calc .calc-label{display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;}
#anya-calc .calc-label b{font-weight:600;}
#anya-calc .calc-pair{display:flex;align-items:center;gap:10px;}
#anya-calc input[type=range]{flex:1;accent-color:var(--brand);}
#anya-calc input[type=number]{width:90px;padding:4px 6px;font:inherit;font-size:13px;color:var(--ink);
  border:1px solid var(--surface);border-radius:6px;background:#fff;transition:border-color .2s ease;}
#anya-calc input[type=number]:focus{outline:none;border-color:var(--brand);}
#anya-calc .calc-result{background:var(--surface);border-radius:var(--radius);padding:14px 16px;margin-top:14px;}
#anya-calc .calc-main{font-size:26px;font-weight:700;color:var(--brand);}
#anya-calc .calc-detail{font-size:12px;opacity:.8;margin-top:4px;}
#anya-calc .calc-formula{font-size:11px;opacity:.6;margin-top:8px;}
</style>
<div id="anya-calc">
  <h3 id="calc-title"></h3>
  <div id="calc-inputs"></div>
  <div class="calc-result">
    <div class="calc-main" id="calc-main"></div>
    <div class="calc-detail" id="calc-detail"></div>
  </div>
  <div class="calc-formula" id="calc-formula"></div>
</div>
<!-- ══ 資料區：以下由模型替換 ══ -->
<script>
const DATA = {
  title: "房貸月付金試算",
  formula: "月付金 = 本金 × r × (1+r)^n ÷ ((1+r)^n − 1)，r = 月利率，n = 總月數（本息平均攤還）",
  // inputs：2-4 個參數
  inputs: [
    { id: "principal", label: "貸款本金", min: 100, max: 3000, step: 10,   def: 800, unit: "萬元" },
    { id: "rate",      label: "年利率",   min: 1,   max: 6,    step: 0.05, def: 2.1, unit: "%" },
    { id: "years",     label: "貸款年期", min: 5,   max: 40,   step: 1,    def: 30,  unit: "年" }
  ],
  // compute：模型改這裡。v 是 {id: 數值}，回傳 { main: 主結果字串, detail: 帶入值說明 }
  compute: function (v) {
    const P = v.principal * 10000, r = v.rate / 100 / 12, n = v.years * 12;
    const m = r === 0 ? P / n : P * r * Math.pow(1 + r, n) / (Math.pow(1 + r, n) - 1);
    return {
      main: "NT$ " + Math.round(m).toLocaleString("zh-TW") + " ／月",
      detail: "本金 " + v.principal + " 萬 × 年利率 " + v.rate + "% × " + v.years + " 年（共 " + n + " 期）"
    };
  }
};
</script>
<!-- ══ 資料區結束 ══ -->
<script>
(function () {
  document.getElementById("calc-title").textContent = DATA.title;
  document.getElementById("calc-formula").textContent = "公式：" + DATA.formula;
  const box = document.getElementById("calc-inputs");
  const values = {};
  DATA.inputs.forEach(function (p) {
    values[p.id] = p.def;
    const row = document.createElement("div");
    row.className = "calc-row";
    const label = document.createElement("div");
    label.className = "calc-label";
    const name = document.createElement("b");
    name.textContent = p.label;
    const unit = document.createElement("span");
    unit.textContent = p.unit;
    label.appendChild(name); label.appendChild(unit);
    const pair = document.createElement("div");
    pair.className = "calc-pair";
    const slider = document.createElement("input");
    slider.type = "range";
    const num = document.createElement("input");
    num.type = "number";
    [slider, num].forEach(function (el) {
      el.min = p.min; el.max = p.max; el.step = p.step; el.value = p.def;
    });
    slider.addEventListener("input", function () {
      num.value = slider.value; values[p.id] = Number(slider.value); recalc();
    });
    num.addEventListener("input", function () {
      const x = Math.min(p.max, Math.max(p.min, Number(num.value) || p.min));
      slider.value = x; values[p.id] = x; recalc();
    });
    pair.appendChild(slider); pair.appendChild(num);
    row.appendChild(label); row.appendChild(pair);
    box.appendChild(row);
  });
  function recalc() {
    const out = DATA.compute(values);
    document.getElementById("calc-main").textContent = out.main;
    document.getElementById("calc-detail").textContent = out.detail;
  }
  recalc();
})();
</script>""",
    },

    # ------------------------------------------------------------------
    "widget_source_browser": {
        "description": "研究來源瀏覽器：證據等級與主題標籤雙重篩選，卡片式來源清單（可外連）。",
        "content": r"""用途：研究來源探索（證據等級 I-VII 篩選 + 主題標籤多選，卡片清單）。
可改：僅「資料區」內的 const DATA — title、sources 陣列（title/url/snippet/level（"I"-"VII"）/topics 標籤陣列）。
不可動：資料區以外的 HTML 結構、CSS 與 JS 邏輯。
註：href 外部網址是唯一允許的外部 URL（使用者點擊才開新分頁，非資源載入）。
height 建議：依來源數估 420-700。

<style>
#anya-src{--brand:#AD4746;--bg:#FFF6F7;--surface:#FFDFE0;--ink:#4B3832;--radius:10px;
  font-family:"SF Pro Rounded",-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif;
  background:var(--bg);color:var(--ink);padding:16px;border-radius:var(--radius);}
#anya-src h3{margin:0 0 4px;font-size:17px;}
#anya-src .src-count{font-size:12px;opacity:.75;margin-bottom:10px;}
#anya-src .src-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;}
#anya-src .src-chip{background:transparent;border:1px solid var(--surface);border-radius:999px;
  padding:3px 10px;font-size:12px;cursor:pointer;transition:all .2s ease;user-select:none;}
#anya-src .src-chip.sel{background:var(--brand);border-color:var(--brand);color:#fff;}
#anya-src .src-card{background:#fff;border:1px solid var(--surface);border-radius:var(--radius);
  padding:12px 14px;margin-top:10px;transition:opacity .2s ease;}
#anya-src .src-card.hide{display:none;}
#anya-src .src-head{display:flex;align-items:baseline;gap:8px;}
#anya-src .src-card a{color:var(--brand);font-weight:600;font-size:14px;text-decoration:none;}
#anya-src .src-card a:hover{text-decoration:underline;}
#anya-src .src-badge{flex:none;font-size:11px;font-weight:700;color:#fff;border-radius:6px;padding:1px 7px;}
#anya-src .lv-high{background:var(--brand);}
#anya-src .lv-mid{background:#C98685;}
#anya-src .lv-low{background:#A79A95;}
#anya-src .src-snippet{font-size:13px;margin:6px 0;line-height:1.5;}
#anya-src .src-tags{font-size:11px;opacity:.7;}
#anya-src .src-tags span{background:var(--surface);border-radius:6px;padding:1px 6px;margin-right:4px;}
</style>
<div id="anya-src">
  <h3 id="src-title"></h3>
  <div class="src-count" id="src-count"></div>
  <div class="src-chips" id="src-levels"></div>
  <div class="src-chips" id="src-topics"></div>
  <div id="src-list"></div>
</div>
<!-- ══ 資料區：以下由模型替換 ══ -->
<script>
const DATA = {
  title: "間歇性斷食研究來源",
  // sources：level 為證據等級 "I"-"VII"（I 最高：系統性回顧/統合分析；VII 最低：專家意見）
  sources: [
    { title: "間歇性斷食對代謝指標影響之統合分析", url: "https://pubmed.ncbi.nlm.nih.gov/00000001/",
      snippet: "納入 28 篇 RCT，顯示 16:8 斷食可小幅降低體重與空腹胰島素，證據品質中等。",
      level: "I", topics: ["代謝", "體重管理"] },
    { title: "限時進食與第二型糖尿病風險：世代研究", url: "https://pubmed.ncbi.nlm.nih.gov/00000002/",
      snippet: "追蹤 12 年的前瞻性世代研究，發現進食時窗小於 10 小時者糖尿病風險略低。",
      level: "IV", topics: ["代謝", "糖尿病"] },
    { title: "斷食期間運動表現變化的隨機對照試驗", url: "https://pubmed.ncbi.nlm.nih.gov/00000003/",
      snippet: "40 名受試者交叉設計，空腹訓練時高強度輸出下降約 5%，耐力表現無顯著差異。",
      level: "II", topics: ["運動表現"] },
    { title: "臨床營養師對斷食飲食法的實務建議", url: "https://www.example.org/expert-opinion-fasting",
      snippet: "專家意見文章，整理常見執行障礙與副作用，建議孕婦與糖尿病患者先諮詢醫師。",
      level: "VII", topics: ["實務建議"] },
    { title: "間歇性斷食與睡眠品質：橫斷面調查", url: "https://pubmed.ncbi.nlm.nih.gov/00000004/",
      snippet: "1,200 名成人問卷調查，晚間進食時窗結束較早者自評睡眠品質較佳。",
      level: "VI", topics: ["睡眠", "實務建議"] }
  ]
};
</script>
<!-- ══ 資料區結束 ══ -->
<script>
(function () {
  const ROMAN = { I: 1, II: 2, III: 3, IV: 4, V: 5, VI: 6, VII: 7 };
  const BANDS = [["全部", 1, 7], ["I-III", 1, 3], ["IV-V", 4, 5], ["VI-VII", 6, 7]];
  let band = BANDS[0];
  const selTopics = new Set();
  document.getElementById("src-title").textContent = DATA.title;
  function badgeClass(lv) {
    const n = ROMAN[lv] || 7;
    return n <= 3 ? "lv-high" : n <= 5 ? "lv-mid" : "lv-low";
  }
  const levelBox = document.getElementById("src-levels");
  BANDS.forEach(function (b, i) {
    const chip = document.createElement("span");
    chip.className = "src-chip" + (i === 0 ? " sel" : "");
    chip.textContent = b[0];
    chip.addEventListener("click", function () {
      band = b;
      levelBox.querySelectorAll(".src-chip").forEach(function (c) { c.classList.remove("sel"); });
      chip.classList.add("sel");
      render();
    });
    levelBox.appendChild(chip);
  });
  const topicBox = document.getElementById("src-topics");
  const allTopics = [];
  DATA.sources.forEach(function (s) {
    s.topics.forEach(function (t) { if (allTopics.indexOf(t) < 0) allTopics.push(t); });
  });
  allTopics.forEach(function (t) {
    const chip = document.createElement("span");
    chip.className = "src-chip";
    chip.textContent = "# " + t;
    chip.addEventListener("click", function () {
      if (selTopics.has(t)) selTopics.delete(t); else selTopics.add(t);
      chip.classList.toggle("sel", selTopics.has(t));
      render();
    });
    topicBox.appendChild(chip);
  });
  const list = document.getElementById("src-list");
  const cards = DATA.sources.map(function (s) {
    const card = document.createElement("div");
    card.className = "src-card";
    const head = document.createElement("div");
    head.className = "src-head";
    const badge = document.createElement("span");
    badge.className = "src-badge " + badgeClass(s.level);
    badge.textContent = "等級 " + s.level;
    const a = document.createElement("a");
    a.href = s.url; a.target = "_blank"; a.rel = "noopener";
    a.textContent = s.title;
    head.appendChild(badge); head.appendChild(a);
    const sn = document.createElement("div");
    sn.className = "src-snippet"; sn.textContent = s.snippet;
    const tags = document.createElement("div");
    tags.className = "src-tags";
    s.topics.forEach(function (t) {
      const sp = document.createElement("span"); sp.textContent = t; tags.appendChild(sp);
    });
    card.appendChild(head); card.appendChild(sn); card.appendChild(tags);
    list.appendChild(card);
    return card;
  });
  function render() {
    let shown = 0;
    DATA.sources.forEach(function (s, i) {
      const n = ROMAN[s.level] || 7;
      const okLevel = n >= band[1] && n <= band[2];
      const okTopic = selTopics.size === 0 ||
        s.topics.some(function (t) { return selTopics.has(t); });
      const ok = okLevel && okTopic;
      cards[i].classList.toggle("hide", !ok);
      if (ok) shown++;
    });
    document.getElementById("src-count").textContent =
      "符合 " + shown + "/" + DATA.sources.length + " 筆";
  }
  render();
})();
</script>""",
    },

    # ------------------------------------------------------------------
    "widget_flashcards": {
        "description": "抽認卡：點卡翻面、上一張/下一張、會了/再看自評（會了即移出輪播），支援鍵盤操作。",
        "content": r"""用途：從文件重點學習的抽認卡（翻面、自評輪播、鍵盤操作）。
可改：僅「資料區」內的 const DATA — title、cards 陣列（front 正面題目、back 背面答案），3-20 張。
不可動：資料區以外的 HTML 結構、CSS 與 JS 邏輯。
鍵盤：空白鍵翻面、← → 切換（需先點一下 widget 取得焦點）。
height 建議：420-480。

<style>
#anya-fc{--brand:#AD4746;--bg:#FFF6F7;--surface:#FFDFE0;--ink:#4B3832;--radius:10px;
  font-family:"SF Pro Rounded",-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif;
  background:var(--bg);color:var(--ink);padding:16px;border-radius:var(--radius);}
#anya-fc h3{margin:0 0 4px;font-size:17px;}
#anya-fc .fc-count{font-size:12px;opacity:.75;margin-bottom:10px;}
#anya-fc .fc-stage{perspective:900px;height:190px;}
#anya-fc .fc-card{position:relative;width:100%;height:100%;cursor:pointer;
  transform-style:preserve-3d;transition:transform .25s ease;}
#anya-fc .fc-card.flip{transform:rotateY(180deg);}
#anya-fc .fc-face{position:absolute;inset:0;backface-visibility:hidden;-webkit-backface-visibility:hidden;
  display:flex;align-items:center;justify-content:center;text-align:center;
  padding:18px 22px;border-radius:var(--radius);font-size:16px;line-height:1.6;}
#anya-fc .fc-front{background:#fff;border:1px solid var(--surface);font-weight:600;}
#anya-fc .fc-back{background:var(--surface);transform:rotateY(180deg);}
#anya-fc .fc-btns{display:flex;gap:8px;margin-top:12px;}
#anya-fc button{flex:1;padding:8px 0;font:inherit;font-size:13px;color:var(--ink);cursor:pointer;
  background:#fff;border:1px solid var(--surface);border-radius:var(--radius);transition:all .2s ease;}
#anya-fc button:hover{border-color:var(--brand);}
#anya-fc button.pri{background:var(--brand);border-color:var(--brand);color:#fff;font-weight:600;}
#anya-fc .fc-done{display:none;text-align:center;padding:48px 0;font-size:18px;font-weight:600;}
#anya-fc .fc-hint{font-size:11px;opacity:.6;margin-top:8px;text-align:center;}
</style>
<div id="anya-fc" tabindex="0">
  <h3 id="fc-title"></h3>
  <div class="fc-count" id="fc-count"></div>
  <div id="fc-play">
    <div class="fc-stage">
      <div class="fc-card" id="fc-card">
        <div class="fc-face fc-front" id="fc-front"></div>
        <div class="fc-face fc-back" id="fc-back"></div>
      </div>
    </div>
    <div class="fc-btns">
      <button id="fc-prev">← 上一張</button>
      <button id="fc-again">再看</button>
      <button id="fc-got" class="pri">會了 ✓</button>
      <button id="fc-next">下一張 →</button>
    </div>
    <div class="fc-hint">點卡片翻面｜空白鍵翻面、← → 切換</div>
  </div>
  <div class="fc-done" id="fc-done">🎉 全部會了！太棒了</div>
</div>
<!-- ══ 資料區：以下由模型替換 ══ -->
<script>
const DATA = {
  title: "統計學重點抽認卡",
  // cards：front 正面題目、back 背面答案
  cards: [
    { front: "p 值的正確定義是什麼？", back: "在虛無假設為真的前提下，觀察到目前資料（或更極端結果）的機率；不是「假設為真的機率」。" },
    { front: "型一錯誤 vs. 型二錯誤？", back: "型一：虛無假設為真卻拒絕它（誤報）；型二：虛無假設為假卻沒拒絕（漏報）。" },
    { front: "信賴區間 95% 的意思？", back: "重複抽樣建構區間，長期而言約 95% 的區間會涵蓋真實母體參數。" },
    { front: "相關 ≠ 因果，為什麼？", back: "可能存在干擾變數、反向因果或巧合；需實驗設計或因果推論方法才能下因果結論。" },
    { front: "中央極限定理說了什麼？", back: "樣本數夠大時，樣本平均數的抽樣分配近似常態，不論母體原本的分配為何。" }
  ]
};
</script>
<!-- ══ 資料區結束 ══ -->
<script>
(function () {
  document.getElementById("fc-title").textContent = DATA.title;
  let pool = DATA.cards.map(function (_, i) { return i; });
  let pos = 0, flipped = false;
  const card = document.getElementById("fc-card");
  function show() {
    if (pool.length === 0) {
      document.getElementById("fc-play").style.display = "none";
      document.getElementById("fc-done").style.display = "block";
      document.getElementById("fc-count").textContent = "剩 0 張";
      return;
    }
    const c = DATA.cards[pool[pos]];
    flipped = false;
    card.classList.remove("flip");
    document.getElementById("fc-front").textContent = c.front;
    document.getElementById("fc-back").textContent = c.back;
    document.getElementById("fc-count").textContent =
      "剩 " + pool.length + " 張｜第 " + (pos + 1) + "/" + pool.length + " 張";
  }
  function flip() {
    flipped = !flipped;
    card.classList.toggle("flip", flipped);
  }
  function move(d) {
    if (pool.length === 0) return;
    pos = (pos + d + pool.length) % pool.length;
    show();
  }
  card.addEventListener("click", flip);
  document.getElementById("fc-prev").addEventListener("click", function () { move(-1); });
  document.getElementById("fc-next").addEventListener("click", function () { move(1); });
  document.getElementById("fc-again").addEventListener("click", function () { move(1); });
  document.getElementById("fc-got").addEventListener("click", function () {
    if (pool.length === 0) return;
    pool.splice(pos, 1);
    if (pos >= pool.length) pos = 0;
    show();
  });
  document.addEventListener("keydown", function (e) {
    if (e.code === "Space") { e.preventDefault(); flip(); }
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "ArrowRight") move(1);
  });
  show();
})();
</script>""",
    },
}
