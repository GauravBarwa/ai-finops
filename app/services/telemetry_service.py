"""Telemetry persistence service — stores execution records as JSONL files.

This is intentionally simple (file-based).  It can be replaced with
PostgreSQL or another store once the need is justified.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.comparison import CompareResponse
from app.schemas.telemetry import TelemetryRecord

logger = get_logger("telemetry_service")


class TelemetryService:
    """Append-only JSONL storage for telemetry records.

    Usage::

        svc = TelemetryService()
        svc.save_comparison(compare_response)
        records = svc.load_records()
    """

    def __init__(self, results_dir: Optional[Path] = None) -> None:
        self._results_dir = results_dir or get_settings().results_path
        self._results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _telemetry_file(self) -> Path:
        """Current telemetry JSONL file (one per day)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._results_dir / f"telemetry_{today}.jsonl"

    def save_record(self, record: TelemetryRecord) -> None:
        """Append a single telemetry record to the current day's file."""
        line = record.model_dump_json() + "\n"
        with open(self._telemetry_file, "a") as f:
            f.write(line)

    def save_comparison(self, response: CompareResponse, source: str = "compare") -> None:
        """Persist all results from a ``CompareResponse`` as telemetry records."""
        for result in response.results:
            record = TelemetryRecord(
                task_id=response.task_id,
                model_id=result.model_id,
                provider=result.provider,
                task_type=response.task_type,
                prompt=response.prompt,
                response=result.response,
                success=result.success,
                error=result.error,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                latency_ms=result.latency_ms,
                estimated_cost=result.estimated_cost,
                source=source,
            )
            self.save_record(record)

        logger.info(
            "Saved %d telemetry records for task_id=%s",
            len(response.results), response.task_id,
        )

    def load_records(self, date: Optional[str] = None) -> list[TelemetryRecord]:
        """Load telemetry records from a specific date (default: today).

        Parameters
        ----------
        date:
            Date string in ``YYYY-MM-DD`` format.  Defaults to today (UTC).
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        filepath = self._results_dir / f"telemetry_{date}.jsonl"
        if not filepath.exists():
            return []

        records: list[TelemetryRecord] = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(TelemetryRecord.model_validate_json(line))
        return records

    def list_dates(self) -> list[str]:
        """Return all dates that have telemetry files."""
        dates: list[str] = []
        for path in sorted(self._results_dir.glob("telemetry_*.jsonl")):
            # telemetry_2026-09-20.jsonl -> 2026-09-20
            date_part = path.stem.replace("telemetry_", "")
            dates.append(date_part)
        return dates
