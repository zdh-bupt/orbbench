from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_TEST_MARKER = PROJECT_ROOT / "reports/observable_regime_demo/FINAL_TEST_ACCESS.json"
FINAL_TEST_TOKEN_ENV = "SATLIFE_FINAL_TEST_TOKEN"


def require_active_final_test_access(expected_out: Path | None = None) -> dict[str, object]:
    if not FINAL_TEST_MARKER.is_file():
        raise PermissionError("Frozen test access requires the one-shot final-test reservation")
    payload = json.loads(FINAL_TEST_MARKER.read_text(encoding="utf-8"))
    if payload.get("status") != "STARTED":
        raise PermissionError(f"Frozen test access is not active: status={payload.get('status')!r}")
    expected_token = str(payload.get("access_token", ""))
    actual_token = os.environ.get(FINAL_TEST_TOKEN_ENV, "")
    if not expected_token or actual_token != expected_token:
        raise PermissionError("Frozen test access token is absent or does not match the active reservation")
    if expected_out is not None:
        reserved_out = Path(str(payload.get("out", ""))).resolve()
        if reserved_out != expected_out.resolve():
            raise PermissionError(
                f"Final-test output does not match reservation: {expected_out} != {reserved_out}"
            )
    return payload
