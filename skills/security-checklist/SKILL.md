---
name: security-checklist
description: >
  Use this skill when code involves security-sensitive operations: database access,
  user authentication, file I/O, network requests, environment variables or secrets,
  external API calls, user-supplied input, eval/exec usage, or when the user asks
  to review code for security issues. Also use when building login flows or handling
  sensitive data.
---

# security-checklist

## Overview

This skill provides a security review checklist and secure coding patterns for Python.
Always reference this skill when writing or reviewing code that touches external data,
credentials, databases, files, or network.

---

## 1. Secrets & 環境變數

```python
# ✅ 正確：從環境變數或 st.secrets 讀取
import os
API_KEY = os.environ["OPENAI_API_KEY"]           # 不存在時直接拋錯（明確）
DB_URL  = os.environ.get("DATABASE_URL", "")     # 有預設值

# Streamlit 專用
API_KEY = st.secrets["OPENAI_API_KEY"]

# ❌ 禁止：hardcode 在程式碼裡
API_KEY = "sk-abc123..."  # 絕對不能這樣做

# ❌ 禁止：print 或 log 出來
print(f"Using key: {API_KEY}")
logger.debug("API key: %s", API_KEY)
```

規則：
- 所有 secret 都放 `.env` / `secrets.toml`，並加入 `.gitignore`
- 永遠不要把 secret 印出來（包括 debug log）
- CI/CD 用環境變數注入，不要把 secret 放進 Docker image

---

## 2. SQL Injection 防護

```python
# ✅ 正確：永遠用參數化查詢
cursor.execute(
    "SELECT * FROM users WHERE email = %s AND active = %s",
    (email, True)
)

# SQLAlchemy ORM（最安全）
user = session.query(User).filter(User.email == email).first()

# SQLAlchemy text()（需要時）
from sqlalchemy import text
result = session.execute(
    text("SELECT * FROM users WHERE email = :email"),
    {"email": email}
)

# ❌ 禁止：字串拼接
query = f"SELECT * FROM users WHERE email = '{email}'"  # SQL Injection!
cursor.execute("SELECT * FROM users WHERE name = '" + name + "'")
```

---

## 3. 路徑穿越（Path Traversal）防護

```python
from pathlib import Path

# ✅ 正確：限制在允許的目錄內
ALLOWED_DIR = Path("/data/uploads").resolve()

def safe_read(filename: str) -> bytes:
    target = (ALLOWED_DIR / filename).resolve()
    # 確認 target 在 ALLOWED_DIR 內
    if not str(target).startswith(str(ALLOWED_DIR)):
        raise ValueError(f"路徑不允許：{filename}")
    return target.read_bytes()

# ❌ 禁止：直接使用使用者輸入的路徑
def unsafe_read(filename: str) -> bytes:
    return open(filename, "rb").read()   # ../../etc/passwd
```

---

## 4. eval / exec 禁止清單

```python
# ❌ 絕對禁止：執行任意程式碼
eval(user_input)
exec(user_input)
__import__(user_input)
subprocess.run(user_input, shell=True)  # shell=True 很危險

# ✅ 替代方案
# 數學計算 → 用 ast.literal_eval 或 sympy
import ast
ast.literal_eval("{'key': 1}")   # 只解析字面值，安全

# 動態執行指令 → 用白名單
ALLOWED_COMMANDS = {"ls", "pwd", "whoami"}
def run_safe(cmd: str, args: list[str]) -> str:
    if cmd not in ALLOWED_COMMANDS:
        raise ValueError(f"不允許的指令：{cmd}")
    return subprocess.run([cmd, *args], capture_output=True, text=True).stdout
```

---

## 5. 輸入驗證

```python
# ✅ 正確：驗證使用者輸入
import re

def validate_email(email: str) -> str:
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError(f"無效的 email：{email}")
    return email.lower().strip()

# 長度限制
def validate_name(name: str) -> str:
    name = name.strip()
    if not 1 <= len(name) <= 100:
        raise ValueError("名稱長度須在 1–100 字元")
    return name

# Pydantic（推薦用於 API 輸入）
from pydantic import BaseModel, EmailStr

class UserInput(BaseModel):
    email: EmailStr
    name: str
    age: int

    class Config:
        max_anystr_length = 200
```

---

## 6. 檔案上傳安全

```python
# ✅ 白名單副檔名
ALLOWED_TYPES = {".pdf", ".png", ".jpg", ".jpeg", ".csv"}

def validate_upload(filename: str, data: bytes) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_TYPES:
        raise ValueError(f"不支援的檔案類型：{ext}")

    # 大小限制（10MB）
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("檔案過大（上限 10MB）")

    # 不要相信 filename，重新生成安全的名稱
    safe_name = f"{uuid.uuid4().hex}{ext}"
    return safe_name
```

---

## 7. 網路請求安全

```python
import httpx

# ✅ 正確：驗證 SSL、設定 timeout、不跟隨惡意 redirect
async def safe_fetch(url: str) -> str:
    # 白名單 domain
    allowed = {"api.openai.com", "mcp.context7.com"}
    from urllib.parse import urlparse
    if urlparse(url).hostname not in allowed:
        raise ValueError(f"不允許的 domain：{url}")

    async with httpx.AsyncClient(
        verify=True,       # ✅ 驗證 SSL
        follow_redirects=False,  # ✅ 不自動跟隨 redirect
        timeout=30.0,      # ✅ 設定 timeout
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text
```

---

## 8. Security Review 快速清單

在 review 或提交程式碼前，確認以下所有項目：

- [ ] 所有 secret 都從環境變數或 `st.secrets` 讀取，沒有 hardcode
- [ ] SQL 查詢全部參數化，沒有字串拼接
- [ ] 使用者輸入的路徑有做邊界檢查
- [ ] 沒有使用 `eval()` / `exec()` 處理使用者輸入
- [ ] 檔案上傳有白名單驗證副檔名和大小限制
- [ ] 對外 HTTP 請求有 timeout 和 SSL 驗證
- [ ] 錯誤訊息不洩漏內部路徑或 stack trace 給使用者
- [ ] 日誌不記錄任何 secret 或個人資料
