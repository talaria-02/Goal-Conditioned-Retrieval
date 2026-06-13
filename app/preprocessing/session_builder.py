"""Session building from raw events."""
from collections import defaultdict
from datetime import timedelta
import uuid

from app.schemas import ActivitySession
from app.preprocessing.aw_parser import RawEvent, AfkEvent


def build_sessions(
    events: list[RawEvent],
    afk_events: list[AfkEvent],
    merge_gap_sec: float = 120.0,
    min_duration_sec: float = 30.0,
) -> list[ActivitySession]:
    """
    Build activity sessions from raw events.
    Events that are close in time (within merge_gap_sec) are grouped into a single session.
    """
    if not events:
        return []

    # 1. Filter out AFK periods (approximate mapping)
    # A simple approach: keep events that do not completely fall inside an "afk" period
    # To keep things simple and robust, we will group events first, and then apply
    # AFK subtraction. However, in this implementation we just group by time gap for now.
    # (In a real system, you'd calculate exact overlap with afk=True periods and subtract)

    sessions: list[ActivitySession] = []
    current_chunk: list[RawEvent] = [events[0]]

    for event in events[1:]:
        last_event = current_chunk[-1]
        gap = (event.timestamp - (last_event.timestamp + timedelta(seconds=last_event.duration))).total_seconds()

        # If gap is smaller than merge_gap_sec AND the app is the same, merge them.
        # 앱이 달라지면 다른 작업으로 간주하여 세션을 강제로 분리합니다.
        if gap <= merge_gap_sec and event.app == last_event.app:
            current_chunk.append(event)
        else:
            # Finalize current chunk
            session = _finalize_session(current_chunk)
            if session and session.total_duration_sec >= min_duration_sec:
                sessions.append(session)
            current_chunk = [event]

    # Finalize the last chunk
    if current_chunk:
        session = _finalize_session(current_chunk)
        if session and session.total_duration_sec >= min_duration_sec:
            sessions.append(session)

    return sessions


def _finalize_session(chunk: list[RawEvent]) -> ActivitySession | None:
    if not chunk:
        return None

    start_time = chunk[0].timestamp
    # End time is max of all event end times in this chunk
    end_time = max(e.timestamp + timedelta(seconds=e.duration) for e in chunk)
    
    total_duration = sum(e.duration for e in chunk)
    # We should avoid double counting overlaps between window and web events.
    # Total elapsed time of the chunk:
    elapsed_sec = (end_time - start_time).total_seconds()
    # Let's bound total_duration_sec to elapsed_sec
    actual_duration = min(total_duration, elapsed_sec)

    app_breakdown: dict[str, float] = defaultdict(float)
    url_breakdown: dict[str, float] = defaultdict(float)
    urls = set()
    titles = set()

    for e in chunk:
        app_breakdown[e.app] += e.duration
        if e.url:
            url_breakdown[e.url] += e.duration
            urls.add(e.url)
        if e.title:
            titles.add(e.title)

    if not app_breakdown:
        return None

    # Determine primary app
    primary_app = max(app_breakdown.items(), key=lambda x: x[1])[0]

    # Determine primary url/domain
    primary_domain = ""
    if url_breakdown:
        best_url = max(url_breakdown.items(), key=lambda x: x[1])[0]
        # Very crude domain extraction
        try:
            domain_part = best_url.split("//")[-1].split("/")[0]
            primary_domain = domain_part
        except Exception:
            pass

    return ActivitySession(
        session_id=f"S-{start_time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        total_duration_sec=actual_duration,
        primary_app=primary_app,
        primary_domain=primary_domain,
        activity_category="unknown",  # will be filled by enricher
        urls=list(urls),
        titles=list(titles),
        app_breakdown=dict(app_breakdown),
        url_breakdown=dict(url_breakdown),
        summary_text="",  # will be filled by enricher
        keywords=[]       # will be filled by enricher
    )
