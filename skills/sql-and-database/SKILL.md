---
name: sql-and-database
description: >
  Use this skill when code involves database operations: SQLAlchemy ORM or Core,
  raw SQL queries, database connection management, migrations, async database access,
  connection pooling, transactions, or when the user asks about database design
  with PostgreSQL, SQLite, or MySQL. Also use for pandas DataFrame to/from database workflows.
---

# sql-and-database

## Overview

This skill provides patterns for safe, efficient database access in Python. Always
reference this skill when writing code that reads from or writes to any database.
Combine with the `security-checklist` skill for SQL injection prevention.

---

## 1. SQLAlchemy 連線設定（推薦架構）

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
import streamlit as st

# ── 取得連線字串（從 secrets，不要 hardcode）──
def _get_db_url() -> str:
    return st.secrets["DATABASE_URL"]
    # 格式範例：
    # PostgreSQL: postgresql+psycopg2://user:pass@host/dbname
    # SQLite:     sqlite:///./local.db
    # MySQL:      mysql+pymysql://user:pass@host/dbname

# ── 建立 Engine（用 @st.cache_resource 確保單例）──
@st.cache_resource
def get_engine():
    url = _get_db_url()
    return create_engine(
        url,
        pool_size=5,          # 連線池大小
        max_overflow=10,      # 超出 pool_size 時最多額外開幾條
        pool_timeout=30,      # 等待連線的逾時（秒）
        pool_recycle=1800,    # 每 30 分鐘回收連線（避免 MySQL 8 小時斷線）
        echo=False,           # True 時印出所有 SQL（開發用）
    )

# SQLite（開發用，不需要 pool）
@st.cache_resource
def get_sqlite_engine():
    return create_engine(
        "sqlite:///./app.db",
        connect_args={"check_same_thread": False},  # SQLite 多執行緒需要
    )

# ── Session Factory ──
Session = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)

# ── 使用 Session（用 context manager 確保關閉）──
def get_users() -> list[dict]:
    with Session() as session:          # 自動 commit + close
        users = session.query(User).filter(User.active == True).all()
        return [{"id": u.id, "name": u.name} for u in users]
```

---

## 2. ORM Model 定義

```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime, timezone

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    name       = Column(String(100), nullable=False)
    active     = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # 關聯（一對多）
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"

class Post(Base):
    __tablename__ = "posts"

    id        = Column(Integer, primary_key=True)
    title     = Column(String(200), nullable=False)
    user_id   = Column(Integer, ForeignKey("users.id"), nullable=False)
    author    = relationship("User", back_populates="posts")

# 建立所有表格（開發用，生產環境改用 Alembic 做遷移）
Base.metadata.create_all(get_engine())
```

---

## 3. CRUD 操作模式

```python
# ── Create ──
def create_user(email: str, name: str) -> User:
    with Session() as session:
        user = User(email=email, name=name)
        session.add(user)
        session.commit()
        session.refresh(user)   # 取得資料庫生成的 id
        return user

# ── Read（單筆）──
def get_user_by_email(email: str) -> User | None:
    with Session() as session:
        return session.query(User).filter(User.email == email).first()

# ── Read（多筆 + 分頁）──
def list_users(page: int = 1, size: int = 20) -> list[User]:
    with Session() as session:
        return (
            session.query(User)
            .filter(User.active == True)
            .order_by(User.created_at.desc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )

# ── Update ──
def update_user_name(user_id: int, name: str) -> bool:
    with Session() as session:
        user = session.get(User, user_id)   # 比 .query().filter().first() 更快
        if not user:
            return False
        user.name = name
        session.commit()
        return True

# ── Delete（軟刪除，建議）──
def deactivate_user(user_id: int) -> bool:
    with Session() as session:
        user = session.get(User, user_id)
        if not user:
            return False
        user.active = False
        session.commit()
        return True
```

---

## 4. 參數化查詢（Raw SQL）

```python
from sqlalchemy import text

# ✅ 正確：永遠用 :param 佔位符
def search_users(keyword: str) -> list[dict]:
    with Session() as session:
        result = session.execute(
            text("SELECT id, name, email FROM users WHERE name ILIKE :kw"),
            {"kw": f"%{keyword}%"}   # 使用者輸入放在 dict，不拼接
        )
        return [dict(row) for row in result.mappings()]

# ❌ 禁止
def bad_search(keyword: str):
    session.execute(text(f"SELECT * FROM users WHERE name LIKE '%{keyword}%'"))
```

---

## 5. Pandas ↔ Database

```python
import pandas as pd

# DataFrame → Database（bulk insert）
def save_dataframe(df: pd.DataFrame, table: str) -> int:
    engine = get_engine()
    rows = df.to_sql(
        name=table,
        con=engine,
        if_exists="append",   # "replace" 會刪表重建，"append" 追加
        index=False,
        chunksize=1000,       # 每次 insert 1000 筆，避免太大的 transaction
        method="multi",       # 使用 multi-row INSERT（較快）
    )
    return rows or 0

# Database → DataFrame
def load_dataframe(table: str, where: str = "") -> pd.DataFrame:
    engine = get_engine()
    query = f"SELECT * FROM {table}"  # noqa: S608
    if where:
        query += f" WHERE {where}"
    return pd.read_sql(query, con=engine)
```

---

## 6. 非同步 DB（async SQLAlchemy）

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# 注意：async 需要 async driver
# PostgreSQL: postgresql+asyncpg://...
# SQLite:     sqlite+aiosqlite:///./app.db
async_engine = create_async_engine(
    st.secrets["ASYNC_DATABASE_URL"],
    echo=False,
)
AsyncSession = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

async def get_user_async(user_id: int) -> User | None:
    async with AsyncSession() as session:
        return await session.get(User, user_id)
```

---

## 7. 常見陷阱

| 問題 | 說明 | 解法 |
|------|------|------|
| Lazy loading N+1 | 迴圈裡每次存取關聯都觸發一次 SQL | 用 `.options(joinedload(User.posts))` 預載 |
| Session 不關閉 | 連線池耗盡 | 永遠用 `with Session() as s:` |
| 跨執行緒共用 Session | SQLAlchemy Session 非執行緒安全 | 每個執行緒/請求建立獨立 Session |
| `expire_on_commit` | commit 後存取物件屬性觸發新查詢 | 設 `expire_on_commit=False` 或在 session 關閉前讀取 |
| SQLite 多執行緒 | 預設禁止跨執行緒使用 | `connect_args={"check_same_thread": False}` |
