"""ActivityWatch data preprocessing modules."""
from app.preprocessing.aw_parser import parse_window_csv, parse_web_csv, parse_afk_csv, load_all_events
from app.preprocessing.session_builder import build_sessions
from app.preprocessing.session_enricher import enrich_session
from app.preprocessing.privacy_filter import PrivacyFilter
from pathlib import Path

def process_aw_data(window_csv, web_csv, afk_csv):
    """Convenience function to run the full preprocessing pipeline."""
    events, afk_events = load_all_events(window_csv, web_csv, afk_csv)
    
    # 1. Privacy Filtering
    blacklist_path = Path(".cache/privacy_blacklist.json")
    privacy_filter = PrivacyFilter(blacklist_path)
    safe_events, safe_afk_events = privacy_filter.filter_events_and_afk(events, afk_events)
    
    # 2. Session Building
    raw_sessions = build_sessions(safe_events, safe_afk_events)
    
    # 3. Enriching
    enriched_sessions = [enrich_session(s) for s in raw_sessions]
    return enriched_sessions
