import os
import argparse
from dotenv import load_dotenv
from pathlib import Path
from app.preprocessing.api_fetcher import fetch_recent_events
from app.preprocessing.privacy_filter import PrivacyFilter
from app.preprocessing.session_builder import build_sessions
from app.preprocessing.session_enricher import enrich_session
from app.retrieval.user_rule_store import UserRuleStore
from app.retrieval.goal_session_matcher import GoalSessionMatcher
from app.schemas import ResearchGoal
from app.llm.coach import RealtimeCoach
from app.notifier import show_toast_notification
from app.retrieval.session_history import SessionHistoryStore
from app.retrieval.dense_retriever import DenseRetriever

def run_coach_pipeline(hours: float = 1.0):
    print("="*50)
    print(f"🚀 실시간 AI 코치 파이프라인 가동 (최근 {hours}시간)")
    
    # 1. Fetch (API에서 실시간 로드)
    raw_events, afk_events = fetch_recent_events(hours=hours)
    if not raw_events:
        print("최근 수집된 활동 데이터가 없습니다.")
        return

    # 2. Preprocess (Privacy Filter -> Session Build -> Enrich)
    blacklist_path = Path(".cache/privacy_blacklist.json")
    privacy_filter = PrivacyFilter(blacklist_path)
    safe_events, safe_afk_events = privacy_filter.filter_events_and_afk(raw_events, afk_events)
    
    raw_sessions = build_sessions(safe_events, safe_afk_events)
    sessions = [enrich_session(s) for s in raw_sessions]
    
    total_min = sum(s.total_duration_sec for s in sessions) / 60.0

    # 3. Match Goals
    rule_store = UserRuleStore(Path(".cache/user_rules.json"))
    goals = rule_store.get_all_goals()
    
    # 목표가 단 하나도 등록되어 있지 않은 초기 상태라면 기본값 세팅
    if not goals:
        default_goal1 = ResearchGoal(goal_id="G1", user_id="u1", title="ROS2 로봇 프로젝트 완성", related_domains=["github.com"], related_keywords=["ros2", "gazebo"])
        default_goal2 = ResearchGoal(goal_id="G2", user_id="u1", title="AI 최적화 기법 과제", related_domains=["cyber.gachon.ac.kr"], related_keywords=["LMS", "최적화"])
        rule_store.add_goal(default_goal1)
        rule_store.add_goal(default_goal2)
        goals = [default_goal1, default_goal2]

    matcher = GoalSessionMatcher(rule_store)
    
    # 백그라운드이므로 interactive=False 로 설정 (질문 없이 바로 넘어감)
    results, unmatched = matcher.match_sessions(goals, sessions, interactive=False)
    
    # 4. 통계 추출 (Deep Work & Healthcare 추가)
    goal_stats = {}
    matched_min = 0
    for goal in goals:
        matched_sessions = results[goal.goal_id]
        dur_min = sum(m.session.total_duration_sec for m in matched_sessions) / 60.0
        
        # 💡 [아이디어 1] 딥워크: 가장 길게 연속으로 집중한 단일 세션 시간
        max_continuous_min = 0
        if matched_sessions:
            max_continuous_min = max(m.session.total_duration_sec for m in matched_sessions) / 60.0
            
        goal_stats[goal.title] = {
            "total_min": dur_min,
            "max_continuous_min": max_continuous_min,
            "session_count": len(matched_sessions) # 끊긴 횟수 파악용
        }
        matched_min += dur_min
        
    # 기타 활동 시간 계산
    goal_stats["기타/휴식(매칭안됨)"] = {
        "total_min": total_min - matched_min,
        "max_continuous_min": 0,
        "session_count": 0
    }

    # 💡 [아이디어 2] 헬스케어: 연속 작업에 따른 휴식(AFK) 분석
    total_afk_min = sum(afk.duration for afk in safe_afk_events) / 60.0
    max_single_afk_min = 0
    if safe_afk_events:
        max_single_afk_min = max(afk.duration for afk in safe_afk_events) / 60.0
        
    health_stats = {
        "total_afk_min": total_afk_min,
        "max_single_afk_min": max_single_afk_min
    }

    print(f"📊 [통계 요약] 총 {total_min:.1f}분 중 목표에 집중한 시간: {matched_min:.1f}분")
    print(f"🩺 [헬스케어] 총 휴식 시간: {total_afk_min:.1f}분 (가장 길게 쉰 시간: {max_single_afk_min:.1f}분)")

    # 5. [RAG] 세션 히스토리 저장 & 과거 유사 세션 검색
    history_store = SessionHistoryStore(Path(".cache/session_history.json"))
    history_store.save_sessions(sessions)  # 오늘 세션 누적 저장
    
    past_context = ""
    past_logs = history_store.load_as_research_logs()
    
    if len(past_logs) > 5:  # 과거 데이터가 충분히 쌍여야 검색 의미가 있음
        try:
            retriever = DenseRetriever()
            retriever.index(past_logs)
            
            # 각 목표별로 과거 유사 세션 검색
            past_lines = []
            for goal in goals:
                candidates = retriever.retrieve(goal.query_text, top_n=3)
                for c in candidates:
                    dur = c.log.metadata.get('duration_sec', 0) / 60.0
                    past_lines.append(
                        f"  - [{c.log.date}] {c.log.title} ({dur:.1f}분, 유사도: {c.dense_score:.2f})"
                    )
            if past_lines:
                past_context = "\n".join(past_lines)
                print(f"📚 [RAG] 과거 유사 세션 {len(past_lines)}개 발견")
        except Exception as e:
            print(f"RAG 검색 스킵 (Embedding 초기화 실패): {e}")

    # 6. LLM Coach 분석 및 알림
    try:
        coach = RealtimeCoach()
        feedback = coach.generate_feedback(goal_stats, total_min, health_stats, past_context)
        print(f"\n💡 [AI 코치 피드백]\n{feedback}\n")
        
        show_toast_notification("🤖 실시간 AI 코치", feedback, duration=15)
        
    except Exception as e:
        print(f"LLM 연동 오류: {e}")
        print("GEMINI_API_KEY 환경변수가 설정되어 있는지 확인하세요.")

if __name__ == "__main__":
    # .env 파일에서 환경변수 로드
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=1.0, help="분석할 최근 N시간")
    parser.add_argument("--loop", action="store_true", help="지정된 간격으로 무한 반복 실행 (데몬 모드)")
    parser.add_argument("--interval", type=int, default=5, help="반복 실행 간격 (분 단위, 기본 5분)")
    args = parser.parse_args()
    
    if args.loop:
        import time
        print(f"🔄 실시간 AI 코치 데몬 모드 시작 (실행 주기: {args.interval}분)")
        while True:
            try:
                run_coach_pipeline(hours=args.hours)
            except Exception as e:
                print(f"❌ 파이프라인 실행 중 오류 발생: {e}")
            
            print(f"⏳ 다음 분석까지 {args.interval}분 대기 중...")
            time.sleep(args.interval * 60)
    else:
        run_coach_pipeline(hours=args.hours)
