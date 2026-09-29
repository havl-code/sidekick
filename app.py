"""Web app for the Sidekick: a FastAPI backend serving a plain HTML/CSS/JS frontend.

Run from this folder with:  uv run app.py
Then open http://127.0.0.1:7860 (it opens automatically).

Each browser tab gets its own Sidekick (its own browser window and MCP servers), keyed by a
session id. The frontend polls /state while a turn runs so the plan and activity update live.
"""

import asyncio
import os
import uuid
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sidekick import SANDBOX, Sidekick, user_entry

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
HOST, PORT = "127.0.0.1", 7860


class Session:
    """One Sidekick plus the conversation it is having. Only one piece of work runs at a time;
    it runs as a task so the user can stop it."""

    def __init__(self):
        self.sidekick = Sidekick()
        self.history = []
        self.task: asyncio.Task | None = None
        self.stop_requested = False

    @property
    def busy(self):
        return self.task is not None and not self.task.done()

    def state(self):
        return {
            "history": self.history,
            "paused": self.sidekick.paused,
            "todos": self.sidekick.todos,
            "activity": self.sidekick.activity,
            "busy": self.busy,
        }


sessions: dict[str, Session] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for session in sessions.values():
        session.sidekick.cleanup()
    sessions.clear()


app = FastAPI(title="Sidekick", lifespan=lifespan)


class TurnRequest(BaseModel):
    message: str
    success_criteria: str = ""


class DecisionRequest(BaseModel):
    approve: bool
    note: str = ""


def get_session(session_id: str) -> Session:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "This session has expired. Reload the page to start a new one.")
    return session


async def run_work(session: Session, work, request: dict | None = None):
    """Run one piece of agent work as a stoppable task, and report failures cleanly.
    request is the user's new entry, if this work starts a turn."""
    if session.busy:
        raise HTTPException(409, "The Sidekick is still working on the last request.")
    session.stop_requested = False
    session.task = asyncio.create_task(work())
    try:
        session.history = await session.task
    except asyncio.CancelledError:
        if not session.stop_requested:
            raise
        session.history = session.sidekick.after_stop(session.history, request)
    except Exception as error:
        raise HTTPException(500, f"The Sidekick hit an error: {error}") from error
    return session.state()


@app.post("/api/sessions")
async def create_session():
    session = Session()
    try:
        await session.sidekick.setup()
    except Exception as error:
        session.sidekick.cleanup()
        raise HTTPException(500, f"Could not start the Sidekick's tools: {error}") from error
    session_id = str(uuid.uuid4())
    sessions[session_id] = session
    return {"session_id": session_id}


@app.post("/api/sessions/{session_id}/turn")
async def turn(session_id: str, request: TurnRequest):
    session = get_session(session_id)
    message = request.message.strip()
    criteria = request.success_criteria.strip()
    if not message:
        raise HTTPException(400, "Type a request for the Sidekick first.")
    if session.sidekick.paused:
        raise HTTPException(409, "Approve or decline the pending action first.")
    return await run_work(
        session,
        lambda: session.sidekick.run_turn(message, criteria, session.history),
        user_entry(message, criteria),
    )


@app.post("/api/sessions/{session_id}/decision")
async def decision(session_id: str, request: DecisionRequest):
    session = get_session(session_id)
    if not session.sidekick.paused:
        raise HTTPException(400, "There is nothing waiting for your decision.")
    return await run_work(
        session, lambda: session.sidekick.resume(session.history, request.approve, request.note.strip())
    )


@app.post("/api/sessions/{session_id}/stop")
async def stop(session_id: str):
    session = get_session(session_id)
    if session.busy:
        session.stop_requested = True
        session.task.cancel()
    return {"stopping": session.stop_requested}


@app.get("/api/sessions/{session_id}/state")
async def state(session_id: str):
    return get_session(session_id).state()


@app.post("/api/sessions/{session_id}/close")
async def close(session_id: str):
    """Called by the page as it unloads (via sendBeacon), so the browser window and MCP servers shut down."""
    session = sessions.pop(session_id, None)
    if session:
        if session.busy:
            session.task.cancel()
        session.sidekick.cleanup()
    return {"closed": session is not None}


@app.get("/api/files")
async def list_files():
    """Everything the Sidekick has written to its sandbox, newest first."""
    files = []
    if os.path.isdir(SANDBOX):
        for folder, _, names in os.walk(SANDBOX):
            for name in names:
                full = os.path.join(folder, name)
                info = os.stat(full)
                files.append({
                    "path": os.path.relpath(full, SANDBOX).replace(os.sep, "/"),
                    "size": info.st_size,
                    "modified": datetime.fromtimestamp(info.st_mtime).isoformat(timespec="seconds"),
                })
    files.sort(key=lambda f: f["modified"], reverse=True)
    return {"files": files}


@app.get("/api/files/{path:path}")
async def download_file(path: str):
    root = os.path.realpath(SANDBOX)
    full = os.path.realpath(os.path.join(root, path))
    if os.path.commonpath([root, full]) != root or not os.path.isfile(full):
        raise HTTPException(404, "That file is not in the sandbox.")
    return FileResponse(full, filename=os.path.basename(full))


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    print(f"Sidekick running at {url}")
    webbrowser.open(url)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
