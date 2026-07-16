"""Transport-level tests for the GR00T 0.6 cached-action protocol."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from insightbench.policies import groot060_client


class _FakeServer:
    def __init__(self, chunk_actions: int):
        self.chunk_actions = chunk_actions
        self.remaining = 0
        self.full_queries = 0
        self.cached_only_queries = 0

    def request(self, method: str, url: str, payload: dict | None = None) -> dict:
        assert method == "POST"
        payload = payload or {}
        if url.endswith("/reset"):
            self.remaining = 0
            return {"ok": True}
        if payload.get("cached_only"):
            self.cached_only_queries += 1
            if self.remaining == 0:
                return {"ok": False, "needs_query": True}
        else:
            assert payload.get("state")
            assert payload.get("images")
            self.full_queries += 1
            self.remaining = self.chunk_actions

        self.remaining -= 1
        return {"ok": True, "action": [[0.0] * 10]}


def _transport_wrapper() -> groot060_client.Groot060ClientWrapper:
    wrapper = groot060_client.Groot060ClientWrapper.__new__(
        groot060_client.Groot060ClientWrapper
    )
    wrapper.endpoint = "http://127.0.0.1:7861/select_action"
    wrapper.reset_endpoint = "http://127.0.0.1:7861/reset"
    wrapper.image_mapping = {"camera": "observation.images.camera"}
    wrapper._supports_cached_only = True
    wrapper._needs_full_query = True
    return wrapper


@pytest.mark.parametrize(
    ("query_freq", "steps", "expected_full_queries"),
    [
        (4, 13, 4),
        (10, 25, 3),
        (12, 25, 5),
    ],
)
def test_cached_transport_handles_arbitrary_query_frequency(
    monkeypatch: pytest.MonkeyPatch,
    query_freq: int,
    steps: int,
    expected_full_queries: int,
) -> None:
    server = _FakeServer(chunk_actions=10)
    monkeypatch.setattr(groot060_client, "_request_json", server.request)
    wrapper = _transport_wrapper()
    state = torch.zeros((2, 16), dtype=torch.float32)
    images = {"observation.images.camera": torch.zeros((2, 3, 8, 8))}

    for step in range(steps):
        if step % query_freq == 0:
            wrapper.reset()
        response = wrapper._request_action_response(state, images, ["task", "task"])
        assert response["ok"]

    assert server.full_queries == expected_full_queries
    assert server.full_queries + server.cached_only_queries >= steps


def test_client_falls_back_to_full_payload_for_old_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _FakeServer(chunk_actions=10)
    monkeypatch.setattr(groot060_client, "_request_json", server.request)
    wrapper = _transport_wrapper()
    wrapper._supports_cached_only = False
    state = torch.zeros((1, 16), dtype=torch.float32)
    images = {"observation.images.camera": torch.zeros((1, 3, 8, 8))}

    for _ in range(3):
        assert wrapper._request_action_response(state, images, ["task"])["ok"]

    assert server.full_queries == 3
    assert server.cached_only_queries == 0


def test_server_cached_only_path_never_requires_observation_payload() -> None:
    server_path = (
        Path.home()
        / "clvla/benchmarks/INSIGHT/my_scripts/eval/groot060_policy_server.py"
    )
    if not server_path.is_file():
        pytest.skip(f"policy server checkout not available: {server_path}")

    spec = importlib.util.spec_from_file_location("groot060_policy_server", server_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    runtime = module.Groot060Runtime(
        checkpoint="unused",
        checkpoint_kind="insight_finetuned",
        lerobot_root="unused",
        embodiment_tag="unused",
        device="cpu",
        state_dim=16,
        action_dim=10,
        camera_keys=["camera"],
        local_files_only=True,
        chunk_actions=2,
        seed=42,
    )
    runtime.load = lambda: None
    runtime.action_chunk = torch.arange(60, dtype=torch.float32).reshape(3, 2, 10)

    first = runtime.select_action({"cached_only": True})
    second = runtime.select_action({"cached_only": True})
    exhausted = runtime.select_action({"cached_only": True})

    assert first["ok"] and first["chunk_index"] == 0
    assert second["ok"] and second["chunk_index"] == 1
    assert exhausted == {
        "ok": False,
        "needs_query": True,
        "chunk_index": 2,
        "chunk_actions": 2,
        "inference_ran": False,
    }
    assert runtime.full_query_count == 0
    assert runtime.cached_only_count == 3


def test_server_resolves_cached_vlm_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_path = (
        Path.home()
        / "clvla/benchmarks/INSIGHT/my_scripts/eval/groot060_policy_server.py"
    )
    if not server_path.is_file():
        pytest.skip(f"policy server checkout not available: {server_path}")

    spec = importlib.util.spec_from_file_location("groot060_policy_server_cache", server_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    repo = tmp_path / "hub/models--nvidia--Cosmos-Reason2-2B"
    snapshot = repo / "snapshots/revision-a"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs/main").write_text("revision-a\n")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (snapshot / name).write_text("{}")

    assert module._resolve_cached_hf_snapshot("nvidia/Cosmos-Reason2-2B") == str(
        snapshot
    )
