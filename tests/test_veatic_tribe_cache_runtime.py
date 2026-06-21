import json
import time

from backend.scripts.run_veatic_tribe_cache import (
    cache_identity_contract,
    claim_is_stale,
    claim_path_for,
    contracts_match,
    is_protected_veatic_cache_write,
    release_claim,
    should_restart_process,
    try_claim_video,
    worker_argv,
)


def test_should_restart_process_after_configured_new_encodes():
    assert should_restart_process(
        encoded_since_restart=25,
        restart_every_n_videos=25,
        remaining_videos=1,
        dry_run=False,
    )


def test_should_not_restart_when_disabled_or_no_remaining_videos():
    assert not should_restart_process(
        encoded_since_restart=25,
        restart_every_n_videos=0,
        remaining_videos=10,
        dry_run=False,
    )
    assert not should_restart_process(
        encoded_since_restart=25,
        restart_every_n_videos=25,
        remaining_videos=0,
        dry_run=False,
    )
    assert not should_restart_process(
        encoded_since_restart=25,
        restart_every_n_videos=25,
        remaining_videos=10,
        dry_run=True,
    )


def test_worker_argv_replaces_workers_and_worker_id():
    argv = [
        "backend/scripts/run_veatic_tribe_cache.py",
        "--workers",
        "3",
        "--worker-id",
        "old",
        "--limit=10",
    ]

    built = worker_argv(argv, 2)

    assert "--workers" in built
    assert built[-4:] == ["--workers", "1", "--worker-id", "2"]
    assert "old" not in built
    assert "--limit=10" in built


def test_claim_video_is_atomic_and_releasable(tmp_path):
    output = tmp_path / "1"
    contract = {"video_num_frames": 64, "restart_every_n_videos": 25}

    claim = try_claim_video(
        output=output,
        video_id="1",
        worker_id="0",
        contract=contract,
        claim_timeout_seconds=3600,
    )

    assert claim is not None
    assert claim_path_for(output).exists()
    assert try_claim_video(
        output=output,
        video_id="1",
        worker_id="1",
        contract=contract,
        claim_timeout_seconds=3600,
    ) is None

    release_claim(output, claim)

    assert not claim_path_for(output).exists()


def test_stale_claim_can_be_reclaimed(tmp_path):
    output = tmp_path / "2"
    output.mkdir()
    claim_path = claim_path_for(output)
    claim_path.write_text(
        json.dumps({"claimed_at_unix": time.time() - 1000, "claim_id": "old"}),
        encoding="utf-8",
    )

    assert claim_is_stale(claim_path, timeout_seconds=1)
    claim = try_claim_video(
        output=output,
        video_id="2",
        worker_id="new",
        contract={"video_num_frames": 64},
        claim_timeout_seconds=1,
    )

    assert claim is not None
    assert claim["worker_id"] == "new"


def test_cache_identity_ignores_operational_runtime_knobs():
    old = {
        "video_num_frames": 64,
        "vjepa21_image_size": 256,
        "restart_every_n_videos": 25,
        "clear_mlx_cache_each_window": False,
    }
    new = {
        "video_num_frames": 64,
        "vjepa21_image_size": 256,
        "restart_every_n_videos": 10,
        "clear_mlx_cache_each_window": True,
    }

    assert cache_identity_contract(old) == {
        "video_num_frames": 64,
        "vjepa21_image_size": 256,
    }
    assert contracts_match(old, new)


def test_mlx_worker_refuses_protected_veatic_cache(monkeypatch, tmp_path):
    import backend.scripts.run_veatic_tribe_cache as cache_runner

    external_root = tmp_path / "external"
    protected = external_root / "benchmarks" / "veatic" / "tribe_cache"
    monkeypatch.setattr(cache_runner, "external_root", lambda: external_root)

    assert is_protected_veatic_cache_write(protected, video_encoder_backend="mlx")
    assert not is_protected_veatic_cache_write(protected, video_encoder_backend="torch")
    assert not is_protected_veatic_cache_write(
        external_root / "benchmarks" / "veatic" / "tribe_cache_mlx",
        video_encoder_backend="mlx",
    )
