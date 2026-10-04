"""Tests for the web app's endpoints that work without a running Sidekick."""

import pytest
from fastapi.testclient import TestClient

import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SANDBOX", str(tmp_path))
    app.sessions.clear()
    yield TestClient(app.app)
    app.sessions.clear()


def test_index_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_unknown_session_is_reported_as_expired(client):
    assert client.get("/api/sessions/nope/state").status_code == 404


def test_empty_request_is_rejected(client):
    app.sessions["s1"] = app.Session()
    response = client.post("/api/sessions/s1/turn", json={"message": "   "})
    assert response.status_code == 400


def test_decision_without_a_pending_action_is_rejected(client):
    app.sessions["s1"] = app.Session()
    assert client.post("/api/sessions/s1/decision", json={"approve": True}).status_code == 400


def test_close_removes_the_session(client):
    app.sessions["s1"] = app.Session()
    assert client.post("/api/sessions/s1/close").json() == {"closed": True}
    assert "s1" not in app.sessions


def test_files_lists_and_downloads_sandbox_files(client, tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "plan.md").write_text("# Plan")

    files = client.get("/api/files").json()["files"]
    assert [f["path"] for f in files] == ["notes/plan.md"]

    response = client.get("/api/files/notes/plan.md")
    assert response.status_code == 200 and response.text == "# Plan"


def test_download_refuses_paths_outside_the_sandbox(client, tmp_path):
    (tmp_path.parent / "secret.txt").write_text("secret")
    assert client.get("/api/files/..%2Fsecret.txt").status_code == 404
    assert client.get("/api/files/missing.txt").status_code == 404
