---
name: research-and-report
description: >
  Use this workflow when the user asks to: research a topic, investigate something,
  analyze documents, compare options, write a report, summarize findings, or any
  multi-step task requiring web search or document analysis. Triggered by keywords
  like "研究", "分析", "比較", "報告", "整理", "research", "analyze", "compare", "report".
---

# Research and Report Workflow

## Overview

This skill handles multi-step research and report-writing tasks. Follow these steps precisely.

---

## Step 1: Save the Request

Use `write_file` to save the user's original request:

```
write_file("/task_request.md", <user's exact task description>)
```

---

## Step 2: Plan with write_todos

Break the task into concrete steps using `write_todos`. Example for a research task:

```
todos:
  - content: "儲存任務請求到 /task_request.md"
    status: completed
  - content: "委派研究：<主題>"
    status: in_progress
  - content: "整合研究結果"
    status: pending
  - content: "撰寫最終報告到 /final_report.md"
    status: pending
  - content: "驗證報告完整性"
    status: pending
```

---

## Step 3: Execute Research

### For web research → Delegate to research-agent

Use the `task()` tool to delegate ONE focused research topic at a time:

```
task(
  subagent_type="research-agent",
  description="<specific research question with context>"
)
```

**Delegation rules:**
- Simple queries → 1 research-agent
- Explicit comparisons (A vs B) → 1 research-agent per subject, run in parallel
- Never break "research X" into sub-topics; let the agent handle comprehensively

### For company internal knowledge → Use company_knowledge_search

```
company_knowledge_search("<query>")
```

### For uploaded documents → Use docstore_search

```
docstore_search("<query>")
```

---

## Step 4: Reflect with think

After each research step, use the `think` tool:

```
think("What did I find? What's still missing? Do I have enough to write the report?")
```

Update todo statuses as steps complete.

---

## Step 5: Write Final Report

Synthesize all findings and write to `/final_report.md`:

```
write_file("/final_report.md", <complete report in markdown>)
```

**Report structure** (choose based on task type):

For research/summary:
```markdown
# [Report Title]

## 摘要
[2-3 sentence overview]

## [Main Section 1]
[Detailed prose paragraphs]

## [Main Section 2]
[Detailed prose paragraphs]

## 結論
[Conclusions and implications]

### 參考來源
[1] Source Title: URL
[2] Source Title: URL
```

**Style rules:**
- Write in paragraph form, not just bullet points
- Cite sources inline: [1], [2], [3]
- No self-referential language ("I found...", "I researched...")
- Minimum 500 words for research reports

---

## Step 6: Verify

Read back the saved request and confirm the report addresses it:

```
read_file("/task_request.md")
```

Update all todos to `completed`. Then provide a brief summary to the user.

---

## Document Analysis Variant

When the task is to analyze uploaded documents (PDF, Word, etc.):

1. Save task to `/task_request.md`
2. Use `docstore_search` with multiple relevant queries
3. Use `think` to organize findings
4. Use `company_knowledge_search` to cross-reference with company knowledge
5. Write analysis to `/analysis_report.md`
6. Verify completeness

---

## Quick Tasks (No Report Needed)

For simple factual questions that don't need a full report:
1. Use `think` to assess if web search is needed
2. Call `company_knowledge_search` or `web_search` directly (via research-agent for web)
3. Respond directly without writing files

Trigger: Task is a single factual question, not a research assignment.
