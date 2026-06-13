from app.preprocessing.api_fetcher import fetch_recent_events
from datetime import datetime

def main():
    print(f"[{datetime.now().isoformat()}] ActivityWatch API 연결 테스트 시작...")
    
    # 최근 24시간 데이터를 가져와 봅니다. (테스트용으로 넉넉하게)
    raw_events, afk_events = fetch_recent_events(hours=24.0)
    
    print(f"가져온 Raw Events (Window/Web): {len(raw_events)} 개")
    print(f"가져온 AFK Events: {len(afk_events)} 개")
    
    if raw_events:
        print("\n[가장 최근 3개 이벤트 미리보기]")
        for e in raw_events[-3:]:
            print(f"  - {e.timestamp.strftime('%H:%M:%S')} | [{e.app}] {e.title[:50]}...")

if __name__ == "__main__":
    main()
