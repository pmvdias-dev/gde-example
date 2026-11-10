import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from flask import Flask, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

AWS_REGION = "us-east-1"
# Opus 5.5 is denied for this account (AccessDeniedException); Haiku 4.5 is enabled.
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SYSTEM_PROMPT = "You are a helpful, concise general-purpose assistant."
# shortcut: only the latest messages are sent as context, older turns are dropped; summarize if long chats need full memory.
MAX_CONTEXT_MESSAGES = 40
MAX_CONTEXT_CHARS = 24000  # ~6k tokens of history; must stay >= MAX_CHARS so the new message always fits
MAX_CHARS = 8000
TITLE_CHARS = 60

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
MAX_EMAIL_CHARS = 254
FREE_MESSAGE_LIMIT = 10
# Paywall is off until a payment system exists; set ENFORCE_FREE_LIMIT=1 to turn it on.
# Even when off, messages_sent is still counted so the limit applies from day one.
ENFORCE_FREE_LIMIT = os.environ.get("ENFORCE_FREE_LIMIT") == "1"
MIN_PASSWORD_CHARS = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT 'free',
    messages_sent INTEGER NOT NULL DEFAULT 0  -- a counter, not COUNT(messages): deleting chats must not reset it
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
"""
# Runs after the user_id migration, since databases from before login lack that column.
USER_INDEX = "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at);"


def load_secret_key(db_path):
    """SECRET_KEY env var, else a key file next to the database shared by all workers."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = Path(db_path).with_name("secret.key")
    try:  # O_EXCL: if two gunicorn workers race on first start, exactly one creates the key
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text().strip()
    key = secrets.token_hex(32)
    with os.fdopen(fd, "w") as f:
        f.write(key)
    return key


app = Flask(__name__)
app.config["DATABASE"] = os.environ.get("DATABASE", str(Path(__file__).with_name("chat.db")))
app.config.update(
    SECRET_KEY=load_secret_key(app.config["DATABASE"]),
    SESSION_COOKIE_HTTPONLY=True,
    # Lax: the cookie is not sent on cross-site POSTs, which blocks CSRF on the login/logout forms.
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE") == "1",  # set to 1 behind HTTPS
)
bedrock = boto3.Session(region_name=AWS_REGION).client("bedrock-runtime")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
        g.db.executescript(SCHEMA)
        ucols = {r["name"] for r in g.db.execute("PRAGMA table_info(users)")}
        if "username" in ucols:
            g.db.execute("ALTER TABLE users RENAME COLUMN username TO email")  # from the first login version
        if "plan" not in ucols:  # users table from before the free limit
            g.db.execute("ALTER TABLE users ADD COLUMN plan TEXT NOT NULL DEFAULT 'free'")
            g.db.execute("ALTER TABLE users ADD COLUMN messages_sent INTEGER NOT NULL DEFAULT 0")
        cols = {r["name"] for r in g.db.execute("PRAGMA table_info(conversations)")}
        if "user_id" not in cols:  # pre-login chats keep user_id NULL and are visible to nobody
            g.db.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE")
        g.db.execute(USER_INDEX)
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def find_conversation(cid):
    return get_db().execute(
        "SELECT id, title FROM conversations WHERE id = ? AND user_id = ?", (cid, g.user["id"])
    ).fetchone()


def make_title(text):
    title = " ".join(text.split())
    return title if len(title) <= TITLE_CHARS else title[: TITLE_CHARS - 1] + "…"


# ---------- Auth ----------
PUBLIC_ENDPOINTS = {"login", "register", "static"}


@app.before_request
def require_login():
    g.user = None
    uid = session.get("user_id")
    if uid is not None:
        g.user = get_db().execute("SELECT id, email, plan, messages_sent FROM users WHERE id = ?", (uid,)).fetchone()
        if g.user is None:  # account no longer exists
            session.clear()
    if g.user is None and request.endpoint not in PUBLIC_ENDPOINTS:
        if request.path.startswith("/api/") or request.path == "/chat":
            return jsonify(error="login required"), 401
        return redirect(url_for("login"))


def start_session(user_id):
    session.clear()  # drop any pre-login session data (session fixation)
    session["user_id"] = user_id


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return redirect(url_for("index")) if g.user else render_template("login.html", mode="login")
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    user = get_db().execute(
        "SELECT id, password_hash FROM users WHERE email = ?", (email,)
    ).fetchone()
    if user is None or not check_password_hash(user["password_hash"], password):
        # Same message for unknown user and wrong password, so registered emails cannot be probed.
        return render_template("login.html", mode="login", error="Invalid email or password.", email=email), 401
    start_session(user["id"])
    return redirect(url_for("index"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return redirect(url_for("index")) if g.user else render_template("login.html", mode="register")
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    error = None
    if len(email) > MAX_EMAIL_CHARS or not EMAIL_RE.fullmatch(email):
        error = "Enter a valid email address."
    elif len(password) < MIN_PASSWORD_CHARS:
        error = f"Password must be at least {MIN_PASSWORD_CHARS} characters."
    if error is None:
        db = get_db()
        try:
            with db:
                uid = db.execute(
                    "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                    (email, generate_password_hash(password), now()),
                ).lastrowid
        except sqlite3.IntegrityError:
            error = "An account with that email already exists."
    if error:
        return render_template("login.html", mode="register", error=error, email=email), 400
    start_session(uid)
    return redirect(url_for("index"))


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- Chat ----------
@app.get("/")
def index():
    return render_template("index.html", email=g.user["email"])


@app.get("/api/conversations")
def list_conversations():
    rows = get_db().execute(
        "SELECT id, title, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
        (g.user["id"],),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/api/conversations/<int:cid>")
def get_conversation(cid):
    conv = find_conversation(cid)
    if conv is None:
        return jsonify(error="conversation not found"), 404
    rows = get_db().execute(
        "SELECT role, text FROM messages WHERE conversation_id = ? ORDER BY id", (cid,)
    ).fetchall()
    return jsonify(id=conv["id"], title=conv["title"], messages=[dict(r) for r in rows])


@app.delete("/api/conversations/<int:cid>")
def delete_conversation(cid):
    db = get_db()
    with db:
        deleted = db.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?", (cid, g.user["id"])
        ).rowcount
    if not deleted:
        return jsonify(error="conversation not found"), 404
    return "", 204


@app.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    cid = body.get("conversation_id")

    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
        return jsonify(error="text must be 1-%d characters" % MAX_CHARS), 400
    text = text.strip()
    if ENFORCE_FREE_LIMIT and g.user["plan"] == "free" and g.user["messages_sent"] >= FREE_MESSAGE_LIMIT:
        return jsonify(error=f"Free limit of {FREE_MESSAGE_LIMIT} messages reached. Upgrade to a paid account to continue."), 402
    if cid is not None and (type(cid) is not int or find_conversation(cid) is None):
        return jsonify(error="conversation not found"), 404

    db = get_db()
    history = []
    if cid is not None:
        rows = db.execute(
            "SELECT role, text FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (cid, MAX_CONTEXT_MESSAGES - 1),
        ).fetchall()
        history = [dict(r) for r in reversed(rows)]
    history.append({"role": "user", "text": text})
    # Drop oldest turns until under the character budget; the new message is always kept.
    while len(history) > 1 and sum(len(m["text"]) for m in history) > MAX_CONTEXT_CHARS:
        history.pop(0)
    if history[0]["role"] != "user":  # Bedrock requires the first turn to be user
        history.pop(0)

    try:
        response = bedrock.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": m["role"], "content": [{"text": m["text"]}]} for m in history],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.5},
        )
    except NoCredentialsError:
        return jsonify(error="AWS credentials not found. Run: aws configure"), 500
    except ClientError as e:
        err = e.response["Error"]
        return jsonify(error=f'{err["Code"]}: {err["Message"]}'), 502

    reply = "".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)
    if not reply.strip():  # an empty assistant turn would make every later Bedrock call fail
        return jsonify(error="Model returned an empty response. Try again."), 502

    # Save the user turn and reply together only after success, so stored roles always alternate.
    ts = now()
    with db:
        db.execute("UPDATE users SET messages_sent = messages_sent + 1 WHERE id = ?", (g.user["id"],))
        if cid is None:
            cid = db.execute(
                "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (g.user["id"], make_title(text), ts, ts),
            ).lastrowid
        else:
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, cid))
        db.executemany(
            "INSERT INTO messages (conversation_id, role, text, created_at) VALUES (?, ?, ?, ?)",
            [(cid, "user", text, ts), (cid, "assistant", reply, ts)],
        )

    return jsonify(conversation_id=cid, title=find_conversation(cid)["title"], reply=reply)


if __name__ == "__main__":
    app.run(debug=True)
