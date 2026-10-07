"""Durable assignments using exclusive files on local or shared storage."""

from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path

from histopia.study._manifest import fingerprint


def _exclusive_json(path: Path, value: dict) -> None:
    """O_EXCL gives each job/artifact/claim a single owner across workers."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


class AssignmentLedger:
    """Immutable assignment, claim and outcome records with unique artifacts.

    No stale claim is inferred from age. An incomplete write also holds its
    reservation until an operator verifies the originating process. This avoids
    depending on SQLite byte-range locks on worker network mounts. A retry is
    an explicitly versioned assignment; previous attempts remain auditable.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        for directory in ("jobs", "artifacts", "claims", "outcomes"):
            (self.path / directory).mkdir(parents=True, exist_ok=True)

    def _path(self, directory: str, key: str) -> Path:
        return self.path / directory / (fingerprint(key) + ".json")

    def add(
        self, job_id: str, artifact_key: str, node: str, payload: dict, *, blocked=False
    ):
        if not all((job_id, artifact_key, node)):
            raise ValueError("job, artifact and node identities are required")
        record = {
            "job_id": job_id,
            "artifact_key": artifact_key,
            "node": node,
            "status": "blocked" if blocked else "pending",
            "payload": payload,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        reservation = self._path("artifacts", artifact_key)
        _exclusive_json(reservation, record)
        try:
            _exclusive_json(self._path("jobs", job_id), record)
        except FileExistsError:
            reservation.unlink()
            raise

    def claim(self, job_id: str, node: str) -> str:
        job = self.read(job_id)
        if job["node"] != node or job["status"] != "pending":
            raise ValueError("job is blocked or assigned elsewhere")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        ticks = (
            Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
        )
        owner = json.dumps(
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "boot_id": boot,
                "start_ticks": ticks,
                "token": uuid.uuid4().hex,
            },
            sort_keys=True,
        )
        try:
            _exclusive_json(
                self._path("claims", job_id),
                {
                    "job_id": job_id,
                    "owner": owner,
                    "status": "running",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except FileExistsError as exc:
            raise ValueError(
                "job is already claimed; inspect its process identity"
            ) from exc
        return owner

    def read(self, job_id: str) -> dict:
        """Read an immutable assignment without its claim or outcome overlays."""
        return json.loads(self._path("jobs", job_id).read_text())

    def finish(self, job_id: str, owner: str, *, status: str, evidence: dict):
        if status not in {"complete", "blocked", "failed"} or not evidence:
            raise ValueError("terminal state requires durable evidence")
        claim = json.loads(self._path("claims", job_id).read_text())
        if claim["owner"] != owner:
            raise ValueError("worker does not own this running job")
        _exclusive_json(
            self._path("outcomes", job_id),
            {
                "job_id": job_id,
                "status": status,
                "evidence": evidence,
                "owner": owner,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def rows(self) -> list[dict]:
        rows = []
        for path in sorted((self.path / "jobs").glob("*.json")):
            row = json.loads(path.read_text())
            for directory in ("claims", "outcomes"):
                state = self._path(directory, row["job_id"])
                if state.exists():
                    try:
                        row.update(json.loads(state.read_text()))
                    except json.JSONDecodeError:
                        row.update(status="incomplete_record_requires_process_audit")
            rows.append(row)
        return sorted(rows, key=lambda row: row["job_id"])
