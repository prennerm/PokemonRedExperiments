from __future__ import annotations

import csv
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


class StatsWriter(ABC):
    """Abstract writer interface for exporting agent statistics."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def write_chunk(self, records: List[Dict], *, final: bool = False) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class JsonStatsWriter(StatsWriter):
    """Write each chunk of stats into a timestamped JSON file."""

    def write_chunk(self, records: List[Dict], *, final: bool = False) -> None:
        if not records:
            return
        suffix = "final" if final else str(int(time.time()))
        file_name = self.base_path / f"stats_{suffix}.json"
        with file_name.open("w", encoding="utf-8") as fh:
            json.dump(
                [_json_serialisable(record) for record in records],
                fh,
                indent=2,
                ensure_ascii=False,
            )

    def close(self) -> None:  # pragma: no cover - nothing to close
        return


class CsvStatsWriter(StatsWriter):
    """Append stats into a CSV file with a stable header."""

    def __init__(self, base_path: Path, file_name: str = "stats.csv") -> None:
        super().__init__(base_path)
        self.file_path = self.base_path / file_name
        self._header_written = self.file_path.exists() and self.file_path.stat().st_size > 0

    def write_chunk(self, records: List[Dict], *, final: bool = False) -> None:
        if not records:
            return
        rows = [_flatten_dict(_json_serialisable(record)) for record in records]
        header = sorted(rows[0].keys())
        with self.file_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            for row in rows:
                writer.writerow(row)

    def close(self) -> None:  # pragma: no cover - nothing persistent to close
        return


def _json_serialisable(record: Dict) -> Dict:
    """Convert numpy scalars to plain Python types for JSON/CSV output."""
    serialised = {}
    for key, value in record.items():
        if isinstance(value, dict):
            serialised[key] = _json_serialisable(value)
        elif isinstance(value, list):
            serialised[key] = [_json_serialisable(item) if isinstance(item, dict) else _convert_value(item) for item in value]
        else:
            serialised[key] = _convert_value(value)
    return serialised


def _convert_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _flatten_dict(data: Dict, prefix: str = "") -> Dict[str, float]:
    """Flatten nested dictionaries using dot notation."""
    items: List[tuple[str, float]] = []
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)
