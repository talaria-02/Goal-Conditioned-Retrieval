import os
import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from app.preprocessing.api_fetcher import fetch_recent_events
from app.preprocessing.privacy_filter import PrivacyFilter
from app.preprocessing.session_builder import build_sessions
from app.preprocessing.session_enricher import enrich_session
from app.retrieval.user_rule_store import UserRuleStore
from app.retrieval.goal_session_matcher import GoalSessionMatcher
from app.schemas import ResearchGoal
from app.llm.coach import RealtimeCoach

# 환경변수 로드
load_dotenv()

st.set_page_config(page_title="AI 행동 코치 대시보드", page_icon="🤖", layout="wide")

st.title("🎯 목표기반 AI 코치: 일일 회고 대시보드")
st.markdown("지난 **24시간** 동안의 나의 활동, 딥워크(연속 집중), 헬스케어 통계를 분석합니다.")

@st.cache_data(ttl=600) # 10분 캐싱
def load_and_process_data():
    raw_events, afk_events = fetch_recent_events(hours=24.0)
    if not raw_events:
        return None, None, None, None, None

    # 전처리 파이프라인
    privacy_filter = PrivacyFilter(Path(".cache/privacy_blacklist.json"))
    safe_events, safe_afk_events = privacy_filter.filter_events_and_afk(raw_events, afk_events)
    
    sessions = build_sessions(safe_events, safe_afk_events)
    sessions = [enrich_session(s) for s in sessions]
    total_min = sum(s.total_duration_sec for s in sessions) / 60.0

    goals = [
        ResearchGoal(goal_id="G1", user_id="u1", title="ROS2 로봇 프로젝트 완성", related_domains=["github.com"], related_apps=[], related_keywords=["ros2", "gazebo"]),
        ResearchGoal(goal_id="G2", user_id="u1", title="AI 최적화 기법 과제", related_domains=["cyber.gachon.ac.kr"], related_apps=[], related_keywords=["LMS", "최적화"])
    ]
    
    rule_store = UserRuleStore(Path(".cache/user_rules.json"))
    matcher = GoalSessionMatcher(rule_store)
    results = matcher.match_sessions(goals, sessions, interactive=False)

    # 통계 추출
    goal_stats = {}
    matched_min = 0
    for goal in goals:
        matched_sessions = results[goal.goal_id]
        dur_min = sum(m.session.total_duration_sec for m in matched_sessions) / 60.0
        max_c_min = max((m.session.total_duration_sec / 60.0 for m in matched_sessions), default=0)
        
        goal_stats[goal.title] = {
            "total_min": dur_min,
            "max_continuous_min": max_c_min,
            "session_count": len(matched_sessions)
        }
        matched_min += dur_min
        
    goal_stats["기타/휴식(매칭안됨)"] = {
        "total_min": total_min - matched_min,
        "max_continuous_min": 0,
        "session_count": 0
    }

    # 헬스케어 통계
    total_afk_min = sum(afk.duration for afk in safe_afk_events) / 60.0
    max_single_afk_min = max((afk.duration / 60.0 for afk in safe_afk_events), default=0)
    health_stats = {
        "total_afk_min": total_afk_min,
        "max_single_afk_min": max_single_afk_min
    }

    return goal_stats, total_min, health_stats, results, goals

with st.spinner("ActivityWatch 서버에서 데이터를 가져와 분석 중입니다..."):
    goal_stats, total_min, health_stats, results, goals = load_and_process_data()

if not goal_stats:
    st.error("지난 24시간 동안 수집된 데이터가 없습니다. ActivityWatch 서버가 켜져 있는지 확인해주세요.")
    st.stop()

# 1. 상단 LLM 코치 피드백
st.subheader("💡 오늘의 AI 종합 평가서")
try:
    coach = RealtimeCoach()
    feedback = coach.generate_feedback(goal_stats, total_min, health_stats)
    st.success(feedback)
except Exception as e:
    st.error(f"AI 코치 피드백 생성 실패 (API KEY를 확인하세요): {e}")

st.divider()

# 2. 핵심 지표 메트릭
col1, col2, col3, col4 = st.columns(4)
col1.metric("⏱️ 총 활동 시간", f"{total_min:.1f} 분")
best_goal = max(goal_stats.items(), key=lambda x: x[1]["total_min"] if x[0] != "기타/휴식(매칭안됨)" else -1)
col2.metric("🏆 가장 많이 한 목표", f"{best_goal[0]}", f"{best_goal[1]['total_min']:.1f} 분")
col3.metric("🔥 딥워크 (최대 연속 집중)", f"{best_goal[1]['max_continuous_min']:.1f} 분", f"{best_goal[1]['session_count']}회 끊김", delta_color="inverse")
col4.metric("🧘 가장 길게 쉰 시간", f"{health_stats['max_single_afk_min']:.1f} 분")

st.divider()

# 3. 목표별 시간 할당 차트
st.subheader("📊 목표별 시간 할당 (Time Allocation)")
chart_data = {
    "목표": [],
    "총 집중 시간 (분)": [],
    "최대 연속 집중 시간 (분)": []
}
for g_title, stats in goal_stats.items():
    chart_data["목표"].append(g_title)
    chart_data["총 집중 시간 (분)"].append(stats["total_min"])
    chart_data["최대 연속 집중 시간 (분)"].append(stats["max_continuous_min"])

df = pd.DataFrame(chart_data).set_index("목표")
st.bar_chart(df)

# 4. 세부 세션 기록
with st.expander("📝 상세 활동 로그 (최근 순)"):
    for goal in goals:
        st.markdown(f"**[{goal.title}]**")
        for m in reversed(results[goal.goal_id][-5:]): # 최근 5개만
            s = m.session
            st.text(f" - {s.start_time.strftime('%H:%M:%S')} ~ {s.end_time.strftime('%H:%M:%S')} | {s.total_duration_sec/60:.1f}분 | 주요 활동: {s.summary_text}")
