"""ActivityWatch CSV parser."""
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RawEvent:
    timestamp: datetime
    duration: float
    app: str = ""
    title: str = ""
    url: str = ""
    source: str = ""  # "window" | "web"


@dataclass
class AfkEvent:
    timestamp: datetime
    duration: float
    status: str  # "afk" | "not-afk"


def parse_datetime(dt_str: str) -> datetime:
    # Handle "2026-06-07T09:22:05.651Z"
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


def parse_window_csv(csv_path: Path) -> list[RawEvent]:
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                RawEvent(
                    timestamp=parse_datetime(row["timestamp"]),
                    duration=float(row.get("duration", 0.0)),
                    app=row.get("app", "unknown"),
                    title=row.get("title", ""),
                    source="window"
                )
            )
    return events


def parse_web_csv(csv_path: Path) -> list[RawEvent]:
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                RawEvent(
                    timestamp=parse_datetime(row["timestamp"]),
                    duration=float(row.get("duration", 0.0)),
                    app="Chrome",  # Web events are typically from a browser
                    title=row.get("title", ""),
                    url=row.get("url", ""),
                    source="web"
                )
            )
    return events


def parse_afk_csv(csv_path: Path) -> list[AfkEvent]:
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                AfkEvent(
                    timestamp=parse_datetime(row["timestamp"]),
                    duration=float(row.get("duration", 0.0)),
                    status=row.get("status", "unknown")
                )
            )
    return events


def load_all_events(
    window_csv: Path, web_csv: Path, afk_csv: Path
) -> tuple[list[RawEvent], list[AfkEvent]]:
    """Load and merge window and web events, sorted by timestamp."""
    window_events = parse_window_csv(window_csv) if window_csv.exists() else []
    web_events = parse_web_csv(web_csv) if web_csv.exists() else []
    
    all_raw = window_events + web_events
    all_raw.sort(key=lambda x: x.timestamp)
    
    afk_events = parse_afk_csv(afk_csv) if afk_csv.exists() else []
    afk_events.sort(key=lambda x: x.timestamp)
    
    return all_raw, afk_events
