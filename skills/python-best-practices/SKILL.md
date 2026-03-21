---
name: python-best-practices
description: >
  Use this skill for tasks involving Python code quality, structure, or correctness.
  Triggers include: writing Python functions or classes, type annotations, error handling,
  async/await patterns, testing, logging, packaging, PEP 8, dataclasses, and general
  Python best practices. Also use when refactoring or reviewing Python code for quality.
---

# python-best-practices

## Overview

This skill provides Python coding standards and patterns. Always reference the relevant
section before writing or reviewing Python code.

---

## 1. Type Annotations（型別標注）

```python
# 基本型別
def greet(name: str, times: int = 1) -> str:
    return (name + "\n") * times

# 複雜型別（Python 3.10+ 可用 X | Y，舊版用 Optional / Union）
from typing import Optional
def find(key: str) -> Optional[str]:
    ...

# 容器型別（Python 3.9+ 可直接用小寫）
def process(items: list[str]) -> dict[str, int]:
    ...

# Callable
from typing import Callable
def apply(fn: Callable[[int], str], value: int) -> str:
    return fn(value)
```

規則：
- 所有公開函式都要有型別標注（參數 + 回傳值）
- 私有函式（`_` 開頭）可選，但建議加
- 不要用裸 `Any`，用 `object` 或更精確的型別

---

## 2. 例外處理（Error Handling）

```python
# ✅ 具體的例外型別
try:
    result = risky_operation()
except ValueError as e:
    logger.warning("Bad input: %s", e)
    raise
except (KeyError, AttributeError) as e:
    logger.error("Data error: %s", e)
    return None

# ❌ 禁止這樣寫
try:
    ...
except Exception:   # 太寬泛，吞掉所有錯誤
    pass

# 自定義例外
class AppError(Exception):
    """應用層錯誤基底類別。"""

class ConfigError(AppError):
    """設定錯誤。"""
```

規則：
- 永遠不要 `except Exception: pass`
- 捕捉具體型別，用 `logger` 記錄後再 `raise` 或回傳安全預設值
- 對外 API 用自定義例外類別，不要暴露底層例外

---

## 3. Dataclass vs TypedDict

```python
from dataclasses import dataclass, field

# 用 dataclass：有行為、需要驗證、或要 mutate
@dataclass
class Config:
    host: str
    port: int = 8080
    tags: list[str] = field(default_factory=list)

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

# 用 TypedDict：純資料容器、要和 dict 互換、或給 JSON schema
from typing import TypedDict

class UserDict(TypedDict):
    id: int
    name: str
    email: str
```

---

## 4. Async / Await

```python
import asyncio
from typing import AsyncGenerator

# 基本 async 函式
async def fetch(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

# 並發執行（同時跑多個任務）
results = await asyncio.gather(
    fetch("https://a.com"),
    fetch("https://b.com"),
)

# Async generator（串流）
async def stream_lines(path: str) -> AsyncGenerator[str, None]:
    async with aiofiles.open(path) as f:
        async for line in f:
            yield line.strip()
```

規則：
- 不要在 async 函式裡呼叫 `time.sleep()`，用 `await asyncio.sleep()`
- 不要在主執行緒裡 `asyncio.run()` 多次，用 `asyncio.get_event_loop().run_until_complete()`
- Streamlit 裡用 `threading.Thread` 跑 async loop，不要直接 `asyncio.run()`

---

## 5. Logging（日誌）

```python
import logging

# 模組層級設定（每個模組都這樣寫）
logger = logging.getLogger(__name__)

# 正確用法
logger.debug("Processing %d items", len(items))    # ✅ % 格式，lazy evaluation
logger.info("Done: %s", result)
logger.warning("Slow query: %.2f ms", elapsed)
logger.error("Failed: %s", exc, exc_info=True)      # exc_info=True 記錄完整 traceback

# 錯誤用法
logger.debug(f"Processing {len(items)} items")      # ❌ f-string 會在 DEBUG 關閉時仍計算
```

---

## 6. 常見 PEP 8 規則

| 規則 | 正確 | 錯誤 |
|------|------|------|
| 縮排 | 4 空格 | Tab |
| 行長 | ≤ 88 字（Black 標準） | > 120 字 |
| 空行 | 函式/類別間 2 行，方法間 1 行 | 不一致 |
| 命名 | `snake_case` 函式/變數，`PascalCase` 類別 | `camelCase` |
| 常數 | `UPPER_SNAKE_CASE` | `lower_case` |
| import | 標準庫 → 第三方 → 本地，各群組空一行 | 混在一起 |

---

## 7. 函式設計原則

- **單一職責**：一個函式只做一件事，超過 30 行考慮拆分
- **純函式優先**：相同輸入永遠相同輸出，避免 side effects
- **早期 return**：減少巢狀 if
- **參數數量**：超過 4 個考慮用 dataclass 打包

```python
# ❌ 深度巢狀
def process(data):
    if data:
        if data.get("key"):
            if data["key"] > 0:
                return data["key"] * 2

# ✅ 早期 return
def process(data):
    if not data:
        return None
    if not data.get("key"):
        return None
    if data["key"] <= 0:
        return None
    return data["key"] * 2
```
