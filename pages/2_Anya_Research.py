import streamlit as st
import os
import json
import re
import time
import traceback
from openai import OpenAI
from datetime import datetime

# --- 1. 必須先定義 parse_reflection ---
def parse_reflection(reflection):
    supplement = []
    pattern = r"章節[：:](.+?)，小標題[：:](.+?)，建議查詢關鍵字[：:](.+)"
    for line in reflection.splitlines():
        m = re.search(pattern, line)
        if m:
            supplement.append({
                "section": m.group(1).strip(),
                "subtitle": m.group(2).strip(),
                "desc": m.group(3).strip()
            })
    return supplement

def split_content_and_sources(content):
    match = re.split(r"## 來源", content, maxsplit=1)
    main = match[0].strip()
    sources = match[1].strip() if len(match) > 1 else ""
    return main, sources

def clean_section_content(section_title, content):
    main, _ = split_content_and_sources(content)
    lines = main.splitlines()
    # 去除重複標題
    while lines and section_title in lines[0]:
        lines = lines[1:]
    # 去除AI自動補充的「很抱歉」等
    lines = [l for l in lines if not l.strip().startswith("很抱歉") and not l.strip().startswith("抱歉")]
    # 去除「來源」、「摘要」
    lines = [l for l in lines if not l.strip().startswith("來源") and not l.strip().startswith("摘要")]
    # 去除「重點整理」、「條列重點」、「小結」及其後所有內容
    for i, l in enumerate(lines):
        if any(key in l for key in ["重點整理", "條列重點", "小結"]):
            lines = lines[:i]
            break
    return "\n".join(lines).strip()

def ddgs_search(query: str, max_retries=3) -> str:
    for attempt in range(max_retries):
        try:
            from ddgs import DDGS
            ddgs = DDGS()
            web_results = ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=3)
            news_results = ddgs.news(query, region="wt-wt", safesearch="moderate", max_results=3)
            all_results = []
            if isinstance(web_results, list):
                all_results.extend(web_results)
            if isinstance(news_results, list):
                all_results.extend(news_results)
            docs = []
            sources = []
            for item in all_results:
                title = item.get("title", "無標題")
                link = item.get("href", "") or item.get("link", "") or item.get("url", "")
                snippet = item.get("body", "") or item.get("snippet", "")
                docs.append(f"- [{title}]({link})\n  > {snippet}")
                if link:
                    sources.append(link)
            if not docs:
                return "No results found."
            markdown_content = "\n".join(docs)
            source_block = "\n\n## 來源\n" + "\n".join(sources)
            return markdown_content + source_block
        except Exception as e:
            if "Ratelimit" in str(e) and attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"Error from DuckDuckGo: {e}"

OPENAI_KEY = st.secrets["OPENAI_KEY"] if "OPENAI_KEY" in st.secrets else os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_KEY)


def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

def section_queries(section_title, section_desc, model="gpt-4.1-mini"):
    current_date = get_current_date()
    prompt = f"""
# Role and Objective
你是一位專業資訊檢索專家，目標是針對下方章節主題，產生**最適合用於網路搜尋的查詢關鍵字**。

# Instructions
- 查詢關鍵字應有助於獲得最新資訊，請考慮目前日期：{current_date}。
- 每個關鍵字必須短、具體、能幫助搜尋到該主題的重點資料。
- 關鍵字不能與章節標題完全相同，也不能只複製章節標題或說明。
- 每個關鍵字都要和其他關鍵字有明顯差異，不可重複。
- 關鍵字要涵蓋不同面向（如：技術、應用、趨勢、挑戰、案例等）。
- 請先思考章節主題的核心概念，再拆解成3個不同面向的查詢關鍵字。
- 請分別用繁體中文與英文各產生3個查詢關鍵字。
- **只回傳 JSON 格式，不要多餘說明或註解。**

# Output Format
{{
    "zh": ["關鍵字1", "關鍵字2", "關鍵字3"],
    "en": ["keyword1", "keyword2", "keyword3"]
}}

# Example 1
章節：「AI在醫療影像診斷的應用」
說明：「探討AI技術如何協助醫療影像判讀與診斷」
回覆：

{{
    "zh": ["醫療影像AI", "深度學習診斷", "醫學影像自動化"],
    "en": ["AI medical imaging", "deep learning diagnosis", "medical image automation"]
}}

# Example 2
章節：「自駕車安全挑戰」
說明：「分析自駕車在安全性上的主要挑戰」
回覆：

{{
    "zh": ["自駕車安全", "自動駕駛風險", "車用感測器失效"],
    "en": ["autonomous vehicle safety", "self-driving risks", "sensor failure in cars"]
}}

# 請針對下方章節產生查詢關鍵字
章節：「{section_title}」
說明：「{section_desc}」
回覆：

"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
        if not content:
            return []
        try:
            queries = json.loads(content)
        except Exception:
            # 嘗試修正格式
            content = re.sub(r"[\u4e00-\u9fff]+：", "", content)
            content = content.replace("'", '"')
            queries = json.loads(content)
        return queries.get("zh", []) + queries.get("en", [])
    except Exception as e:
        print(f"section_queries error: {e}")
        return []

def plan_report(topic, search_summaries, model="o4-mini"):
    prompt = f"""Formatting re-enabled
# Role and Objective
你是一位專業技術寫手，目標是針對「{topic}」這個主題，根據下方搜尋摘要，規劃一份完整、深入、結構化的研究報告。

# Instructions
- 報告最少需包含3個章節，每章節需有明確標題。
- 每個章節**必須包含2-4個小標題**（子議題），每個小標題下要有2-3句細節說明。
- 小標題可涵蓋：產業現況、技術細節、國際比較、未來趨勢、挑戰、解決方案、案例、數據等。
- 章節標題必須簡潔明確，僅能為主題詞，不可為完整句子或段落，不可有「0、」或「摘要」等字樣。
- 報告最後一章必須為「結論」或「總結」，簡要歸納全篇重點與未來展望。
- 請用繁體中文條列式回覆，格式如下：

# Output Format
1. 章節標題
    - 小標題1：細節說明
    - 小標題2：細節說明
    - 小標題3：細節說明
2. 章節標題
    - 小標題1：細節說明
    - 小標題2：細節說明
    - 小標題3：細節說明
...

# 搜尋摘要
{search_summaries}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def is_valid_title(title):
    # 標題不能太長、不能是完整句子、不能有0、不能有多個逗號或句號
    if len(title) > 25: return False
    if title.startswith("0、"): return False
    if "。" in title or "，" in title: return False
    if "摘要" in title or "很抱歉" in title: return False
    return True

def parse_sections(plan: str):
    section_pattern = r"\d+\.\s*([^\n]+)"
    sub_pattern = r"-\s*([^\n：:]+)[：:]\s*([^\n]+)"
    sections = []
    section_blocks = re.split(r"\d+\.\s*", plan)[1:]
    for block in section_blocks:
        lines = block.strip().split("\n")
        title = lines[0].strip()
        if not is_valid_title(title):
            continue  # 跳過不合理標題
        subs = []
        for line in lines[1:]:
            m = re.match(sub_pattern, line.strip())
            if m:
                subs.append({"subtitle": m.group(1).strip(), "desc": m.group(2).strip()})
        sections.append({"title": title, "subtitles": subs})
    return sections

def ensure_conclusion(sections, topic):
    titles = [s["title"] for s in sections]
    if not any("結論" in t or "總結" in t for t in titles):
        # 自動加一個結論章節
        sections.append({"title": "結論", "subtitles": [{"subtitle": "總結", "desc": f"{topic}的重點歸納與未來展望"}]})
    return sections

def section_write(section_title, section_desc, search_results, model="gpt-4.1-mini"):
    prompt = f"""
# Role and Objective
你是一位專業技術寫手，根據下方章節主題、說明與「多筆網路查詢結果」，撰寫一段內容豐富、結構清晰、具體詳實的章節內容。

# Instructions
- 內容需至少50字，並涵蓋：具體數據、真實案例、國際比較、產業現況、技術細節、未來趨勢、挑戰與解決方案。
- 必須根據下方每一筆查詢結果，整合出完整內容。
- 每個小標題下必須有2-3段具體說明。
- 條列重點、重點整理、小結只能出現在全篇最前面，不要在每個章節出現。
- 請勿產生「很抱歉」、「摘要」、「來源」等說明文字，正文只需撰寫章節內容。
- 文末請用「## 來源」列出所有引用來源（Markdown格式），來源必須來自下方查詢結果。
- 請勿省略細節，若有多個觀點請分段說明。
- 請勿重複內容，避免空泛敘述。
- 請用繁體中文撰寫。

# 章節主題
{section_title}

# 章節說明
{section_desc}

# 多筆查詢結果（每筆都要參考）
{search_results}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def reflect_report(report: str, reflect_history=None, model="o4-mini"):
    if reflect_history is None:
        reflect_history = []
    prompt = f"""
你是一位專業審稿人，請檢查下方報告的結構與內容，若有缺漏請用**JSON格式**回覆，格式如下：
[
  {{
    "section": "章節標題",
    "subtitle": "小標題",
    "desc": "需補強的內容說明",
    "suggested_queries": ["查詢關鍵字1", "查詢關鍵字2"]
  }},
  ...
]
若內容已完整，請回覆：OK
# 報告內容
{report}
# 前一輪補強歷程
{json.dumps(reflect_history, ensure_ascii=False)}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def key_takeaway(report: str, model="gpt-4.1-mini"):
    prompt = f"""請根據下方完整報告內容，歸納出全篇最重要的5-8點重點(Key Takeaway)，用繁體中文條列式回覆，不要有多餘說明。
# 報告內容
{report}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- 3. pipeline 主流程 ---
def deep_research_pipeline(topic, max_reflect_rounds=2):
    logs = []
    reflect_history = []
    time_stats = {}
    start_time = time.time()
    progress = st.progress(0)
    step_count = 8 + max_reflect_rounds  # 8個主要步驟+反思補強
    step_now = 0

    t0 = time.time()
    with st.spinner("產生查詢關鍵字...", show_time=True):
        queries = section_queries(topic, topic)
        logs.append({"step": "generate_queries", "queries": queries})
        step_now += 1
        progress.progress(step_now/step_count)
    t1 = time.time()
    time_stats["產生查詢關鍵字"] = t1 - t0

    t0 = time.time()
    all_results = []
    with st.spinner("查詢網路資料...", show_time=True):
        for q in queries:
            result = ddgs_search(q)
            all_results.append(result)
            step_now += 1/len(queries)
            progress.progress(min(step_now/step_count, 1.0))
            time.sleep(0.5)
        logs.append({"step": "search", "results": all_results})
    t1 = time.time()
    time_stats["查詢網路資料"] = t1 - t0

    t0 = time.time()
    with st.spinner("規劃章節...", show_time=True):
        search_summary = "\n\n".join(all_results)
        plan = plan_report(topic, search_summary)
        logs.append({"step": "plan_report", "plan": plan})
        step_now += 1
        progress.progress(step_now/step_count)
    t1 = time.time()
    time_stats["規劃章節"] = t1 - t0

    t0 = time.time()
    with st.spinner("解析章節...", show_time=True):
        sections = parse_sections(plan)
        sections = ensure_conclusion(sections, topic)
        logs.append({"step": "parse_sections", "sections": sections})
        step_now += 1
        progress.progress(step_now/step_count)
    t1 = time.time()
    time_stats["解析章節"] = t1 - t0

    t0 = time.time()
    section_contents = []
    with st.spinner("撰寫章節內容...", show_time=True):
        for section in sections:
            s_queries = []
            for sub in section["subtitles"]:
                sub_queries = section_queries(sub["subtitle"], sub["desc"])
                s_queries.extend(sub_queries)
            s_results = []
            for q in s_queries:
                s_results.append(ddgs_search(q))
                time.sleep(0.5)
            search_results = "\n\n".join(s_results)
            content = section_write(
                section["title"],
                "；".join([f"{sub['subtitle']}：{sub['desc']}" for sub in section["subtitles"]]),
                search_results
            )
            section_contents.append({
                "title": section["title"],
                "content": content
            })
            logs.append({
                "step": "section",
                "section": section["title"],
                "queries": s_queries,
                "search_results": search_results,
                "content": content
            })
        step_now += 1
        progress.progress(step_now/step_count)
    t1 = time.time()
    time_stats["撰寫章節內容"] = t1 - t0

    t0 = time.time()
    report = "\n\n".join([f"## {s['title']}\n\n{s['content']}" for s in section_contents])
    logs.append({"step": "combine_report", "report": report})
    step_now += 1
    progress.progress(step_now/step_count)
    t1 = time.time()
    time_stats["組合報告"] = t1 - t0

    # 多輪反思與自動補強
    for i in range(max_reflect_rounds):
        t0 = time.time()
        with st.spinner(f"反思檢查第{i+1}輪...", show_time=True):
            reflection = reflect_report(report, reflect_history)
            reflect_history.append({"round": i+1, "reflection": reflection})
            logs.append({"step": "reflection", "round": i+1, "reflection": reflection})
            step_now += 1
            progress.progress(min(step_now/step_count, 1.0))
            if reflection.strip().upper() == "OK":
                break
            try:
                supplement_list = json.loads(reflection)
            except Exception:
                # fallback: 舊格式
                supplement_list = parse_reflection(reflection)
            for item in supplement_list:
                section_title = item["section"]
                subtitle = item["subtitle"]
                desc = item.get("desc", "")
                queries = item.get("suggested_queries", [])
                if not queries:
                    queries = section_queries(subtitle, desc)
                new_results = [ddgs_search(q) for q in queries]
                new_content = section_write(subtitle, desc, "\n\n".join(new_results))
                # 進階：補強內容 append 到對應章節/小標題
                for section in section_contents:
                    if section["title"] == section_title:
                        if "supplements" not in section:
                            section["supplements"] = []
                        section["supplements"].append({
                            "subtitle": subtitle,
                            "desc": desc,
                            "content": new_content,
                            "round": i+1,
                            "queries": queries
                        })
                        # 也可直接 append 到原 content
                        section["content"] += f"\n\n### {subtitle}（補強第{i+1}輪）\n{new_content}"
                logs.append({"step": "supplement", "item": item, "new_queries": queries, "new_content": new_content})
            report = "\n\n".join([f"## {s['title']}\n\n{s['content']}" for s in section_contents])
            logs.append({"step": "combine_report", "report": report})
        t1 = time.time()
        time_stats[f"反思補強第{i+1}輪"] = t1 - t0

    # pipeline 結束時進度條100%
    progress.progress(1.0)
    end_time = time.time()
    total_time = end_time - start_time

    # 產生乾淨的完整報告
    #full_report = ""
    #for section in section_contents:
    #    cleaned = clean_section_content(section["title"], section["content"])
    #    if cleaned:
    #        full_report += f"## {section['title']}\n\n{cleaned}\n\n---\n"

    full_report = ""
    for section in section_contents:
        cleaned = clean_section_content(section["title"], section["content"])
        supplement_blocks = []
        if "supplements" in section:
            for sup in section["supplements"]:
                supplement_blocks.append(
                    f"### {sup['subtitle']}（補強第{sup['round']}輪）\n{sup['content']}"
                )
        supplement_text = "\n\n".join(supplement_blocks)
        if cleaned or supplement_text:
            full_report += f"## {section['title']}\n\n{cleaned}\n\n{supplement_text}\n\n---\n"
    
    # 產生全篇重點
    t0 = time.time()
    key_points = key_takeaway(full_report)
    t1 = time.time()
    time_stats["全篇重點產生"] = t1 - t0

    return {
        "topic": topic,
        "plan": plan,
        "sections": section_contents,
        "report": report,
        "full_report": full_report,
        "key_points": key_points,
        "reflection": reflection,
        "reflect_history": reflect_history,
        "logs": logs,
        "total_time": total_time,
        "time_stats": time_stats
    }

# --- 4. Streamlit UI ---
st.set_page_config(page_title="安妮亞的超能力研究報告", layout="wide")
st.title("📝 安妮亞的超能力深度研究大作戰！🌟")

topic = st.text_input("主題")
if st.button("產生完整報告"):
    result = deep_research_pipeline(topic)
    section_contents = result["sections"]
    reflection = result["reflection"]
    reflect_history = result["reflect_history"]
    total_time = result["total_time"]
    full_report = result["full_report"]
    key_points = result["key_points"]
    time_stats = result["time_stats"]

    st.success(f"本次運算共花費 {int(total_time//60)} 分 {int(total_time%60)} 秒")

    tabs = st.tabs(["完整報告", "章節內容", "目錄", "反思歷程", "Debug Logs"])

    # 完整報告Tab
    with tabs[0]:
        st.markdown("### 📝 全篇重點 Key Takeaway")
        st.markdown(key_points)
        st.markdown("---")
        st.markdown("### 📝 完整報告")
        st.markdown(full_report)

    # 章節內容Tab
    with tabs[1]:
        inner_tabs = st.tabs([section['title'] for section in section_contents])
        for i, section in enumerate(section_contents):
            with inner_tabs[i]:
                main, sources = split_content_and_sources(section["content"])
                st.markdown(main)
                # 顯示補強內容
                if "supplements" in section:
                    for sup in section["supplements"]:
                        st.info(f"【補強第{sup['round']}輪】{sup['subtitle']}：{sup['desc']}")
                        st.markdown(sup["content"])
                if sources:
                    with st.expander("來源（點我展開）"):
                        st.markdown("## 來源\n" + sources)

    # 目錄
    with tabs[2]:
        st.markdown("## 目錄")
        for section in section_contents:
            st.markdown(f"- [{section['title']}](#{section['title'].replace(' ', '-')})")

    # 反思歷程
    with tabs[3]:
        st.header("🧐 報告反思與審查")
        for rh in reflect_history:
            reflection = rh["reflection"]
            if reflection.strip().upper() == "OK":
                st.success(f"第{rh['round']}輪反思：報告內容結構完整，無明顯遺漏。")
            else:
                st.warning(f"第{rh['round']}輪反思建議：")
                # 嘗試解析為 JSON
                try:
                    parsed = json.loads(reflection)
                    if isinstance(parsed, list):
                        for idx, item in enumerate(parsed, 1):
                            section = item.get("section", "")
                            subtitle = item.get("subtitle", "")
                            desc = item.get("desc", "")
                            queries = item.get("suggested_queries", [])
                            st.markdown(
                                f"""
    **{idx}. 章節：** {section}  
    **小標題：** {subtitle}  
    **需補強說明：** {desc}  
    **建議查詢關鍵字：** {', '.join(queries)}
    ---
                                """
                            )
                    else:
                        st.markdown(reflection)
                except Exception:
                    # 若不是 JSON，直接顯示原始內容
                    st.markdown(reflection)

    # Debug logs
    with tabs[4]:
        st.markdown("## Debug Logs")
        st.markdown("### 各步驟耗時")
        for k, v in time_stats.items():
            st.write(f"{k}: {v:.1f} 秒")
        st.markdown("---")
        for log in result["logs"]:
            st.markdown(f"**Step:** {log.get('step', '')}")
            if "error" in log:
                st.error(f"Error: {log['error']}")
            if "traceback" in log:
                st.code(log["traceback"], language="python")
            for k, v in log.items():
                if k not in ["step", "error", "traceback"]:
                    st.write(f"{k}: {v}")
            st.markdown("---")
