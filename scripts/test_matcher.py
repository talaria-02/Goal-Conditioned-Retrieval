import argparse
from pathlib import Path
from app.schemas import ResearchGoal
from app.preprocessing import process_aw_data
from app.retrieval.user_rule_store import UserRuleStore
from app.retrieval.goal_session_matcher import GoalSessionMatcher

def find_latest_csv(base_dir: Path, prefix: str) -> Path:
    files = list(base_dir.glob(f"{prefix}*.csv"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=str, help="Path to window CSV")
    parser.add_argument("--web", type=str, help="Path to web CSV")
    parser.add_argument("--afk", type=str, help="Path to afk CSV")
    args = parser.parse_args()

    base_dir = Path(".")
    window_csv = Path(args.window) if args.window else find_latest_csv(base_dir, "aw-events-export-aw-watcher-window")
    web_csv = Path(args.web) if args.web else find_latest_csv(base_dir, "aw-events-export-aw-watcher-web-chrome")
    afk_csv = Path(args.afk) if args.afk else find_latest_csv(base_dir, "aw-events-export-aw-watcher-afk")

    if not all([window_csv, web_csv, afk_csv]):
        print("Missing required CSV files. Please specify them or ensure they exist in the directory.")
        return

    print(f"Using Window CSV: {window_csv.name}")
    print(f"Using Web CSV:    {web_csv.name}")
    print(f"Using AFK CSV:    {afk_csv.name}\n")
    # 1. 전처리 파이프라인 (Phase 1)
    print("데이터를 불러오고 전처리하는 중...")
    sessions = process_aw_data(window_csv, web_csv, afk_csv)
    
    # 2. 사용자 목표 정의 (가상)
    goals = [
        ResearchGoal(
            goal_id="G1", user_id="u1", title="ROS2 로봇 프로젝트 완성", 
            related_domains=["github.com"], related_apps=[],
            related_keywords=["gazebo", "robot", "ros2"] # 의도적으로 키워드를 적게 줌
        ),
        ResearchGoal(
            goal_id="G2", user_id="u1", title="AI 최적화 기법 수업", 
            related_domains=["cyber.gachon.ac.kr"], related_apps=[],
            related_keywords=["LMS"]
        )
    ]
    
    # 3. 매칭 모듈 및 룰 저장소 로드 (Phase 2)
    store_path = base_dir / ".cache" / "user_rules.json"
    rule_store = UserRuleStore(store_path)
    matcher = GoalSessionMatcher(rule_store)
    
    # 4. 매칭 실행 (Interactive 모드)
    results = matcher.match_sessions(goals, sessions, interactive=True)
    
    # 5. 최종 매칭된 통계 결과 출력
    print("\n" + "="*60)
    print("📊 [최종 결과] 목표별 시간 투자 분석")
    print("="*60)
    for goal in goals:
        matched = results[goal.goal_id]
        total_sec = sum(m.session.total_duration_sec for m in matched)
        print(f"\n🎯 목표: {goal.title} (총 투자 시간: {total_sec/60:.1f} 분)")
        for m in matched:
            print(f"  - [{m.match_reason}] {m.session.total_duration_sec/60:.1f}분 | {m.session.summary_text}")

if __name__ == "__main__":
    main()
