from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
BASE_DIR_data = Path("/home/staff_homes/sittardt/projects/MultimodalVideoTopicModelling/")
DATA_ROOT = BASE_DIR_data / "data"
ANNOTATION_ROOT = BASE_DIR_data / "data" / "output" / "annotation" / "cross_video"
DB_PATH = BASE_DIR_data / "data" / "annotation.sqlite3"
STATIC_DIR = BASE_DIR / "static"
print(f"data path: {DATA_ROOT}")
print(f"annotation path: {ANNOTATION_ROOT}")
print(f"static path: {STATIC_DIR}")
LOCK_MINUTES = 30

# --- Lifespan ---
@asynccontextmanager
async def lifespan(_: FastAPI):
    init_schema()
    yield

app = FastAPI(title="Video Topic Annotation", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/data", StaticFiles(directory=DATA_ROOT), name="data")
app.mount("/files", StaticFiles(directory=ANNOTATION_ROOT), name="files")

# --- Helpers ---
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def to_iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None

def from_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None

def rel_file_url(path: str | Path) -> str:
    file_path = Path(path)

    print(f"DEBUG input: {file_path}")
    print(f"DEBUG DATA_ROOT: {DATA_ROOT}")


    if str(file_path).startswith("/data/"):
        file_path = DATA_ROOT / file_path.relative_to("/data")
    elif str(file_path).startswith("/files/"):
        file_path = ANNOTATION_ROOT / file_path.relative_to("/files")
    elif not file_path.is_absolute():
        file_path = ANNOTATION_ROOT / file_path
    file_path = file_path.resolve()
    try:
        return f"/data/{file_path.relative_to(DATA_ROOT).as_posix()}"
    except ValueError:
        try:
            return f"/files/{file_path.relative_to(ANNOTATION_ROOT).as_posix()}"
        except ValueError:
            return file_path.as_posix()


def _is_valid_username(name: str) -> bool:
    if not name:
        return False
    low = name.strip().lower()
    if "test" in low or "home" in low:
        return False
    return True


def _purge_invalid_users(conn: sqlite3.Connection) -> None:
    # Remove users whose username contains 'test' or 'home', and clear task references
    rows = conn.execute("SELECT id, username FROM users WHERE LOWER(username) LIKE '%test%' OR LOWER(username) LIKE '%home%'").fetchall()
    ids = [int(r[0]) for r in rows]
    if not ids:
        return
    placeholders = ",".join(["?"] * len(ids))
    conn.execute(f"UPDATE tasks SET claimed_by_user_id = NULL WHERE claimed_by_user_id IN ({placeholders})", ids)
    conn.execute(f"UPDATE tasks SET completed_by_user_id = NULL WHERE completed_by_user_id IN ({placeholders})", ids)
    conn.execute(f"DELETE FROM users WHERE id IN ({placeholders})", ids)
    conn.commit()

# --- HTML Template ---
def render_shell(title: str = "Video Topic Annotation") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <link rel="stylesheet" href="/static/styles.css" />
</head>
<body>
    <div class="page-shell">
        <header class="topbar">
            <div>
                <div class="eyebrow">Multimodal annotation workspace</div>
                <h1>Video Topic Annotation</h1>
            </div>
            <div class="topbar__actions">
                <span id="userBadge" class="pill pill--muted">Not signed in</span>
                <button id="logoutBtn" class="ghost-btn" type="button" hidden>Log out</button>
            </div>
        </header>

        <main class="layout">
            <section class="panel panel--left">
                <div id="loginCard" class="card">
                    <h2>Sign in</h2>
                    <p>Enter a display name to join the annotation queue.</p>
                    <div class="form-row">
                        <input id="usernameInput" type="text" placeholder="e.g. anna" autocomplete="name" />
                        <button id="loginBtn" class="primary-btn" type="button">Enter workspace</button>
                    </div>
                    <p class="hint">The app stores only a username cookie. No separate accounts are required.</p>
                </div>

                <div id="dashboardCard" class="stack" hidden>
                    <div class="card">
                        <h2>Queue</h2>
                        <div class="queue-actions">
                            <button class="primary-btn" data-claim="image_intrusion" type="button">Claim image intrusion</button>
                            <button class="primary-btn primary-btn--alt" data-claim="topic_matching" type="button">Claim topic matching</button>
                            <button class="ghost-btn" data-claim="any" type="button">Claim any task</button>
                        </div>
                        <div class="stats-grid">
                            <div><span>Open</span><strong id="statOpen">0</strong></div>
                            <div><span>Claimed</span><strong id="statClaimed">0</strong></div>
                            <div><span>Done</span><strong id="statDone">0</strong></div>
                            <div><span>Your active</span><strong id="statActive">0</strong></div>
                        </div>
                    </div>

                    <div class="card">
                        <h2>Workflow</h2>
                        <ol class="workflow">
                            <li>Claim a task from the shared queue.</li>
                            <li>Review the clean topic representation and images.</li>
                            <li>Submit your response and move to the next item.</li>
                        </ol>
                    </div>
                </div>
            </section>

            <section class="panel panel--right">
                <div id="taskEmpty" class="hero card">
                    <h2>Waiting for a task</h2>
                    <p>Sign in, claim a task, and the annotation panel will appear here.</p>
                </div>

                <div id="taskCard" class="card task-card" hidden>
                    <div class="task-header">
                        <div>
                            <div id="taskTypeBadge" class="pill">Task</div>
                            <h2 id="taskTitle">Annotation task</h2>
                        </div>
                        <div class="task-meta">
                            <span id="taskVideo"></span>
                        </div>
                    </div>
                            <div class="topic-section">
                        <div class="topic-label" style="font-weight: bold;">Topic</div>
                        <div id="taskTopic" class="topic-banner" style="background-color: #f0f0f0; border: 1px solid #ccc; padding: 10px;">
                        </div>
                    </div>
                    <div id="taskInstructions" class="instructions"></div>
                    <div id="taskUsage" class="usage-summary"></div>
                    <div id="taskWords" class="word-chips"></div>
                    <div id="taskBody"></div>

                    <div class="response-box" id="responseBox" hidden>
                        <h3>Response</h3>
                        <div id="responseFields"></div>
                        <label class="comment-field">
                            Optional comment
                            <textarea id="commentInput" rows="3" placeholder="Add a note for the project team"></textarea>
                        </label>
                    </div>

                    <div class="task-actions">
                        <button id="submitBtn" class="primary-btn" type="button">Submit</button>
                        <button id="releaseBtn" class="ghost-btn" type="button">Release</button>
                        <button id="nextBtn" class="ghost-btn" type="button" hidden>Claim next</button>
                    </div>

                    <div id="taskMessage" class="message"></div>
                </div>
            </section>
        </main>
    </div>
    <script src="/static/app.js?v=2" defer></script>
</body>
</html>"""

# --- DB ---
@contextmanager
def db_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
    finally:
        conn.close()

def init_schema() -> None:
    with db_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                point_id TEXT NOT NULL,
                task_type TEXT NOT NULL CHECK(task_type IN ('image_intrusion', 'topic_matching')),
                assignment_index INTEGER NOT NULL DEFAULT 1 CHECK(assignment_index IN (1, 2)),
                payload_json TEXT NOT NULL,
                sort_index INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'claimed', 'completed')),
                claimed_by_user_id INTEGER,
                claimed_at TEXT,
                lock_expires_at TEXT,
                completed_by_user_id INTEGER,
                completed_at TEXT,
                response_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(point_id, task_type, assignment_index),
                FOREIGN KEY (claimed_by_user_id) REFERENCES users(id),
                FOREIGN KEY (completed_by_user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status_type_sort ON tasks(status, task_type, sort_index, id);
            CREATE INDEX IF NOT EXISTS idx_tasks_claimed_by ON tasks(claimed_by_user_id, status);
            CREATE INDEX IF NOT EXISTS idx_tasks_point_type ON tasks(point_id, task_type);
        """)
        # Purge invalid users on startup
        _purge_invalid_users(conn)

# --- User ---
def get_user_id(conn: sqlite3.Connection, username: str) -> int:
    normalized = username.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Username is required")
    if not _is_valid_username(normalized):
        raise HTTPException(status_code=400, detail="Invalid username")
    now = to_iso(utcnow())
    row = conn.execute("SELECT id FROM users WHERE username = ?", (normalized,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO users (username, last_seen_at) VALUES (?, ?)", (normalized, now))
        row = conn.execute("SELECT id FROM users WHERE username = ?", (normalized,)).fetchone()
    else:
        conn.execute("UPDATE users SET last_seen_at = ? WHERE id = ?", (now, row["id"]))
    return int(row["id"])

# --- Task Metadata ---
def _load_point_metadata() -> list[dict[str, Any]]:
    points_path = ANNOTATION_ROOT / "annotation_points.json"
    if not points_path.exists():
        raise FileNotFoundError(f"Missing annotation points file: {points_path}")
    with points_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _load_task_metadata(point_id: str, task_type: str) -> dict[str, Any] | None:
    metadata_path = ANNOTATION_ROOT / task_type / point_id / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    images = metadata.get("images", [])
    payload = {
        "point_id": metadata.get("id", point_id),
        "task_type": task_type,
        "video": metadata.get("video"),
        "topic": metadata.get("topic"),
        "topic_name": metadata.get("topic_name"),
        "words": metadata.get("words", []),
        "topic_representation": metadata.get("topic_representation") or metadata.get("topic_name"),
        "representative_docs": metadata.get("representative_docs", []),
        "selected_segments": metadata.get("selected_segments", []),
        "images": [rel_file_url(path) for path in images],
        "instructions": "Select the intruder image that does not belong with the clean topic representation." if task_type == "image_intrusion" else "Judge how well the image set matches the clean topic representation.",
    }
    return payload

# --- Task Management ---
def _release_expired_claims(conn: sqlite3.Connection) -> None:
    now = to_iso(utcnow())
    conn.execute(
        "UPDATE tasks SET status = 'open', claimed_by_user_id = NULL, claimed_at = NULL, lock_expires_at = NULL, updated_at = ? WHERE status = 'claimed' AND lock_expires_at IS NOT NULL AND lock_expires_at < ?",
        (now, now),
    )

def _task_row_to_payload(row: sqlite3.Row, user_id: int | None = None) -> dict[str, Any]:
    payload = json.loads(row["payload_json"])
    payload["images"] = [rel_file_url(image) for image in payload.get("images", [])]
    payload.update({
        "task_db_id": int(row["id"]),
        "status": row["status"],
        "claimed_by_user_id": row["claimed_by_user_id"],
        "claimed_at": row["claimed_at"],
        "lock_expires_at": row["lock_expires_at"],
        "response_json": json.loads(row["response_json"]) if row["response_json"] else None,
        "mine": user_id is not None and row["claimed_by_user_id"] == user_id,
    })
    return payload

def import_annotations() -> dict[str, int]:
    init_schema()
    points = _load_point_metadata()
    inserted, skipped = 0, 0
    with db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for sort_index, point in enumerate(points):
            point_id = str(point["id"])
            for task_type in ("image_intrusion", "topic_matching"):
                payload = _load_task_metadata(point_id, task_type)
                if not payload:
                    skipped += 1
                    continue
                payload_json = json.dumps(payload, ensure_ascii=False)
                for assignment_index in (1, 2):
                    existing = conn.execute(
                        "SELECT id FROM tasks WHERE point_id = ? AND task_type = ? AND assignment_index = ?",
                        (point_id, task_type, assignment_index),
                    ).fetchone()
                    if existing is None:
                        conn.execute(
                            "INSERT INTO tasks (point_id, task_type, assignment_index, payload_json, sort_index, status, updated_at) VALUES (?, ?, ?, ?, ?, 'open', ?)",
                            (point_id, task_type, assignment_index, payload_json, sort_index, to_iso(utcnow())),
                        )
                        inserted += 1
                    else:
                        conn.execute(
                            "UPDATE tasks SET payload_json = ?, sort_index = ?, updated_at = ? WHERE id = ?",
                            (payload_json, sort_index, to_iso(utcnow()), existing["id"]),
                        )
        conn.commit()
    return {"inserted": inserted, "skipped": skipped}

def get_task(task_id: int, username: str | None = None) -> dict[str, Any]:
    with db_conn() as conn:
        user_id = get_user_id(conn, username) if username else None
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        return _task_row_to_payload(row, user_id=user_id)

def claim_next_task(username: str, task_type: str) -> dict[str, Any] | None:
    if task_type not in {"image_intrusion", "topic_matching", "any"}:
        raise HTTPException(status_code=400, detail="Invalid task_type")
    with db_conn() as conn:
        user_id = get_user_id(conn, username)
        now, now_iso, expires_iso = utcnow(), to_iso(utcnow()), to_iso(utcnow() + timedelta(minutes=LOCK_MINUTES))
        conn.execute("BEGIN IMMEDIATE")
        _release_expired_claims(conn)
        active = conn.execute(
            "SELECT * FROM tasks WHERE claimed_by_user_id = ? AND status = 'claimed' AND lock_expires_at IS NOT NULL AND lock_expires_at >= ? ORDER BY claimed_at DESC, id DESC LIMIT 1",
            (user_id, now_iso),
        ).fetchone()
        if active:
            conn.commit()
            return _task_row_to_payload(active, user_id=user_id)
        where_type, params = "", []
        if task_type != "any":
            where_type, params = "AND task_type = ?", [task_type]
        candidate = conn.execute(
            f"SELECT t.* FROM tasks t WHERE t.status = 'open' {where_type} AND NOT EXISTS (SELECT 1 FROM tasks t2 WHERE t2.point_id = t.point_id AND t2.task_type = t.task_type AND t2.assignment_index != t.assignment_index AND (t2.completed_by_user_id = ? OR t2.claimed_by_user_id = ?)) ORDER BY t.sort_index ASC, t.id ASC LIMIT 1",
            [*params, user_id, user_id],
        ).fetchone()
        if not candidate:
            conn.commit()
            return None
        conn.execute(
            "UPDATE tasks SET status = 'claimed', claimed_by_user_id = ?, claimed_at = ?, lock_expires_at = ?, updated_at = ? WHERE id = ?",
            (user_id, now_iso, expires_iso, now_iso, candidate["id"]),
        )
        conn.commit()
        claimed = conn.execute("SELECT * FROM tasks WHERE id = ?", (candidate["id"],)).fetchone()
        return _task_row_to_payload(claimed, user_id=user_id)

def submit_task(task_id: int, username: str, response: dict[str, Any]) -> dict[str, Any]:
    with db_conn() as conn:
        user_id = get_user_id(conn, username)
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        if row["status"] == "completed":
            raise HTTPException(status_code=409, detail="Task is already completed")
        if row["claimed_by_user_id"] not in (None, user_id):
            raise HTTPException(status_code=403, detail="Task is claimed by another annotator")
        # Normalize response for image_intrusion: prefer original index when provided
        if row["task_type"] == "image_intrusion" and isinstance(response, dict):
            orig = response.get("selected_image_original_index")
            if orig is not None:
                try:
                    response["selected_image_index"] = int(orig)
                except (TypeError, ValueError):
                    pass
            elif "selected_image_index" in response:
                try:
                    response["selected_image_index"] = int(response["selected_image_index"])
                except (TypeError, ValueError):
                    pass

        now_iso = to_iso(utcnow())
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE tasks SET status = 'completed', completed_by_user_id = ?, completed_at = ?, response_json = ?, lock_expires_at = NULL, updated_at = ? WHERE id = ?",
            (user_id, now_iso, json.dumps(response, ensure_ascii=False), now_iso, task_id),
        )
        conn.commit()
        completed = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _task_row_to_payload(completed, user_id=user_id)

def release_task(task_id: int, username: str) -> dict[str, Any]:
    with db_conn() as conn:
        user_id = get_user_id(conn, username)
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        if row["claimed_by_user_id"] not in (None, user_id):
            raise HTTPException(status_code=403, detail="Task is claimed by another annotator")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE tasks SET status = 'open', claimed_by_user_id = NULL, claimed_at = NULL, lock_expires_at = NULL, updated_at = ? WHERE id = ?",
            (to_iso(utcnow()), task_id),
        )
        conn.commit()
        released = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _task_row_to_payload(released, user_id=user_id)

def dashboard_state(username: str) -> dict[str, Any]:
    with db_conn() as conn:
        user_id = get_user_id(conn, username)
        _release_expired_claims(conn)
        counts = conn.execute(
            "SELECT SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_count, SUM(CASE WHEN status = 'claimed' THEN 1 ELSE 0 END) AS claimed_count, SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS done_count, SUM(CASE WHEN claimed_by_user_id = ? AND status = 'claimed' THEN 1 ELSE 0 END) AS active_count FROM tasks",
            (user_id,),
        ).fetchone()
        active = conn.execute(
            "SELECT * FROM tasks WHERE claimed_by_user_id = ? AND status = 'claimed' ORDER BY claimed_at DESC, id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return {
            "username": username,
            "counts": {k: int(v or 0) for k, v in {"open": counts["open_count"], "claimed": counts["claimed_count"], "completed": counts["done_count"], "active": counts["active_count"]}.items()},
            "active_task": _task_row_to_payload(active, user_id=user_id) if active else None,
        }

# --- API ---
def _username_from_request(request: Request) -> str | None:
    return request.cookies.get("annotator_name") or request.headers.get("x-annotator-name")

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return render_shell()

@app.get("/api/session")
def api_session(request: Request):
    return {"username": _username_from_request(request)}

@app.post("/api/session")
async def set_session(request: Request):
    payload = await request.json()
    username = str(payload.get("username", "")).strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    with db_conn() as conn:
        get_user_id(conn, username)
    response = JSONResponse({"username": username})
    response.set_cookie("annotator_name", username, httponly=True, samesite="lax")
    return response

@app.post("/api/logout")
def logout() -> Response:
    response = JSONResponse({"ok": True})
    response.delete_cookie("annotator_name")
    return response

@app.get("/api/dashboard")
def api_dashboard(request: Request):
    username = _username_from_request(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not signed in")
    return dashboard_state(username)

@app.post("/api/tasks/claim")
async def api_claim(request: Request):
    username = _username_from_request(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not signed in")
    payload = await request.json()
    task_type = str(payload.get("task_type", "any")).strip()
    task = claim_next_task(username, task_type)
    return {"task": task, "message": "No matching tasks are available" if not task else None}

@app.get("/api/tasks/{task_id}")
def api_task(task_id: int, request: Request):
    username = _username_from_request(request)
    return get_task(task_id, username=username)

@app.post("/api/tasks/{task_id}/submit")
async def api_submit(task_id: int, request: Request):
    username = _username_from_request(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not signed in")
    payload = await request.json()
    response = payload.get("response", {})
    if not isinstance(response, dict):
        raise HTTPException(status_code=400, detail="response must be an object")
    return submit_task(task_id, username, response)

@app.post("/api/tasks/{task_id}/release")
def api_release(task_id: int, request: Request):
    username = _username_from_request(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not signed in")
    return release_task(task_id, username)

@app.get("/api/export")
def api_export():
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT t.*, u.username AS completed_by FROM tasks t LEFT JOIN users u ON u.id = t.completed_by_user_id WHERE t.status = 'completed' ORDER BY t.id ASC"
        ).fetchall()
        completed = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            completed.append({
                "task_id": row["id"],
                "point_id": row["point_id"],
                "task_type": row["task_type"],
                "video": payload.get("video"),
                "topic": payload.get("topic"),
                "topic_name": payload.get("topic_name"),
                "response": json.loads(row["response_json"]) if row["response_json"] else None,
                "completed_by": row["completed_by"],
                "completed_at": row["completed_at"],
            })
    return {"completed": completed}

# --- Main ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the annotation web app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8333)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--init-db", action="store_true", help="Initialize the SQLite database and exit")
    args = parser.parse_args()
    init_schema()
    if args.init_db:
        result = import_annotations()
        print(json.dumps(result, indent=2))
        return
    import uvicorn
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()