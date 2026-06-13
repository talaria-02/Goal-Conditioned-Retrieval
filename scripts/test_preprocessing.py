from pathlib import Path
from app.preprocessing import process_aw_data
from app.preprocessing.aw_parser import load_all_events

def main():
    base_dir = Path(".")
    window_csv = base_dir / "aw-events-export-aw-watcher-window_hermes-Katana-17-B13VFK-2026-06-07.csv"
    web_csv = base_dir / "aw-events-export-aw-watcher-web-chrome_hermes-Katana-17-B13VFK-2026-06-07.csv"
    afk_csv = base_dir / "aw-events-export-aw-watcher-afk_hermes-Katana-17-B13VFK-2026-06-07.csv"

    events, afk_events = load_all_events(window_csv, web_csv, afk_csv)
    print(f"Loaded {len(events)} total window/web events, {len(afk_events)} afk events.")

    sessions = process_aw_data(window_csv, web_csv, afk_csv)
    print(f"\nCreated {len(sessions)} meaningful sessions after preprocessing.")
    print("-" * 50)
    
    for idx, s in enumerate(sessions):
        print(f"Session {idx+1} | {s.activity_category.upper()} | {s.total_duration_sec:.1f} sec")
        print(f"  Time: {s.start_time} ~ {s.end_time}")
        print(f"  Summary: {s.summary_text}")
        print(f"  Keywords: {', '.join(s.keywords[:5])}")
        print("-" * 50)

if __name__ == "__main__":
    main()
