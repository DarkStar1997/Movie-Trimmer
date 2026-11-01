"""On-demand VLM supervisor tests — fake process, fake health probe.

The real llama-server needs ~20GB of VRAM and a model file; these test
the lifecycle logic only (spawn-once, refcount, fail-cleanup, argv).
"""

import pytest

from scanner import vlm


class FakeProc:
    def __init__(self, alive=True):
        self._alive = alive
        self.terminated = False
        self.killed = False
        self.returncode = 0

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return self.returncode


@pytest.fixture(autouse=True)
def reset_state():
    vlm._proc, vlm._refs = None, 0
    yield
    vlm._proc, vlm._refs = None, 0


def _env(monkeypatch):
    monkeypatch.setenv("MODEL", "/models/fake.gguf")
    monkeypatch.setenv("MMPROJ", "/models/fake-mm.gguf")


def test_acquire_spawns_once_release_by_last_ref(monkeypatch):
    _env(monkeypatch)
    spawned = []
    monkeypatch.setattr(vlm.subprocess, "Popen",
                        lambda argv, **kw: (spawned.append(argv)
                                            or FakeProc()))
    monkeypatch.setattr(vlm, "_healthy", lambda timeout=2.0: True)

    vlm.acquire()
    vlm.acquire()                       # second pin, same process
    assert len(spawned) == 1
    assert vlm.status()["running"] and vlm.status()["pinned_by"] == 2

    vlm.release()                       # still pinned by one holder
    assert vlm.status()["running"] and vlm.status()["pinned_by"] == 1
    vlm.release()                       # last -> terminated, VRAM freed
    assert not vlm.status()["running"]


def test_argv_carries_the_entrypoint_knobs(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setenv("CTX", "4096")
    monkeypatch.setenv("PARALLEL", "2")
    monkeypatch.setenv("EXTRA_ARGS", "-ub 1024")
    argv = vlm._argv()
    assert argv[0] == "/opt/llama/llama-server"
    assert "-m" in argv and "/models/fake.gguf" in argv
    assert "--mmproj" in argv and "/models/fake-mm.gguf" in argv
    assert "4096" in argv and "2" in argv and "-ub" in argv


def test_acquire_requires_model_env(monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("MMPROJ", raising=False)
    with pytest.raises(vlm.VlmUnavailable):
        vlm._argv()


def test_acquire_load_failure_raises_and_cleans_up(monkeypatch):
    """Server dies during load: acquire raises, process reaped, refs 0."""
    _env(monkeypatch)
    holder = {}
    monkeypatch.setattr(vlm.subprocess, "Popen",
                        lambda argv, **kw: holder.setdefault("p", FakeProc()))
    monkeypatch.setattr(vlm, "_healthy", lambda timeout=2.0: False)
    monkeypatch.setattr(vlm.config, "VLM_START_TIMEOUT_S", 0.05)
    with pytest.raises(vlm.VlmUnavailable):
        vlm.acquire()
    assert holder["p"].terminated
    st = vlm.status()
    assert not st["running"] and st["pinned_by"] == 0


def test_acquire_server_exit_during_load_raises(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(vlm.subprocess, "Popen",
                        lambda argv, **kw: FakeProc(alive=False))
    with pytest.raises(vlm.VlmUnavailable):
        vlm.acquire()
    assert not vlm.status()["running"]
