"""Fetch events directly from ActivityWatch REST API."""
import requests
from datetime import datetime, timedelta, timezone
from app.preprocessing.aw_parser import RawEvent, AfkEvent

AW_API_BASE = "http://localhost:5600/api/0"

def get_buckets() -> dict:
    """존재하는 모든 버킷 목록을 가져옵니다."""
    try:
        response = requests.get(f"{AW_API_BASE}/buckets")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to connect to ActivityWatch: {e}")
        return {}

def find_bucket_id(buckets: dict, watcher_name: str) -> str | None:
    """주어진 watcher_name(예: aw-watcher-window)을 포함하는 버킷 ID를 찾습니다."""
    for bucket_id, bucket_info in buckets.items():
        if bucket_info.get("client", "").startswith(watcher_name) or watcher_name in bucket_id:
            return bucket_id
    return None

def fetch_events_from_bucket(bucket_id: str, start_time: datetime, end_time: datetime) -> list:
    """특정 버킷에서 시간 범위 내의 이벤트를 가져옵니다."""
    params = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat()
    }
    try:
        response = requests.get(f"{AW_API_BASE}/buckets/{bucket_id}/events", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to fetch events from {bucket_id}: {e}")
        return []

def fetch_recent_events(hours: float = 1.0) -> tuple[list[RawEvent], list[AfkEvent]]:
    """최근 N시간 동안의 이벤트를 AW API에서 직접 가져옵니다."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)
    
    buckets = get_buckets()
    if not buckets:
        return [], []
        
    window_bucket = find_bucket_id(buckets, "aw-watcher-window")
    web_bucket = find_bucket_id(buckets, "aw-watcher-web")
    afk_bucket = find_bucket_id(buckets, "aw-watcher-afk")
    
    raw_events = []
    afk_events = []
    
    # 1. Window Events 파싱
    if window_bucket:
        events = fetch_events_from_bucket(window_bucket, start_time, end_time)
        for e in events:
            raw_events.append(RawEvent(
                timestamp=datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")),
                duration=e["duration"],
                app=e["data"].get("app", "unknown"),
                title=e["data"].get("title", ""),
                source="window"
            ))
            
    # 2. Web Events 파싱
    if web_bucket:
        events = fetch_events_from_bucket(web_bucket, start_time, end_time)
        for e in events:
            raw_events.append(RawEvent(
                timestamp=datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")),
                duration=e["duration"],
                app="Chrome",
                title=e["data"].get("title", ""),
                url=e["data"].get("url", ""),
                source="web"
            ))
            
    # 3. AFK Events 파싱
    if afk_bucket:
        events = fetch_events_from_bucket(afk_bucket, start_time, end_time)
        for e in events:
            afk_events.append(AfkEvent(
                timestamp=datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")),
                duration=e["duration"],
                status=e["data"].get("status", "unknown")
            ))
            
    # 시간순 정렬
    raw_events.sort(key=lambda x: x.timestamp)
    afk_events.sort(key=lambda x: x.timestamp)
    
    return raw_events, afk_events
