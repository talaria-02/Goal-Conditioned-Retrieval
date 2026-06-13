import os
import time
import uuid
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from app.preprocessing.api_fetcher import fetch_recent_events
from app.preprocessing.privacy_filter import PrivacyFilter
from app.preprocessing.session_builder import build_sessions
from app.preprocessing.session_enricher import enrich_session
from app.retrieval.user_rule_store import UserRuleStore
from app.retrieval.goal_session_matcher import GoalSessionMatcher
from app.retrieval.query_understanding import build_query
from app.retrieval.query_expansion import expand_goal_query
from app.schemas import ResearchGoal
from app.llm.coach import RealtimeCoach

# 환경변수 로드
load_dotenv()

st.set_page_config(page_title="AI 행동 코치 대시보드", page_icon="🤖", layout="wide")

st.title("🎯 목표기반 AI 코치: 일일 회고 대시보드")
st.markdown("지난 **24시간** 동안의 나의 활동, 딥워크(연속 집중), 헬스케어 통계를 분석합니다.")

# --- 사이드바: 목표 관리 UI ---
st.sidebar.header("📝 목표(Goal) 관리")
rule_store = UserRuleStore(Path(".cache/user_rules.json"))
current_goals = rule_store.get_all_goals()

# 등록된 목표가 하나도 없을 경우 기본 목표 셋업
if not current_goals:
    default_goal1 = ResearchGoal(goal_id="G1", user_id="u1", title="ROS2 로봇 프로젝트 완성", related_domains=[], related_keywords=[])
    default_goal2 = ResearchGoal(goal_id="G2", user_id="u1", title="AI 최적화 기법 과제", related_domains=[], related_keywords=[])
    rule_store.add_goal(default_goal1)
    rule_store.add_goal(default_goal2)
    current_goals = [default_goal1, default_goal2]

st.sidebar.subheader("현재 등록된 목표")
for g in current_goals:
    with st.sidebar.expander(f"📌 {g.title}"):
        st.caption(f"ID: {g.goal_id}")
        
        # 만약 과거에 등록해둔 수동 룰이 있다면 살짝만 표시
        if g.related_domains:
            st.caption(f"수동 도메인 룰: {', '.join(g.related_domains)}")
        if g.related_keywords:
            st.caption(f"수동 키워드 룰: {', '.join(g.related_keywords)}")
            
        # [NEW] AI가 확장한 키워드 목록 보여주기 (확실하게 expand_goal_query 호출)
        try:
            q_obj = build_query(g)
            exp_result = expand_goal_query(g, q_obj, use_cache=True, use_mock_fallback=False)
            terms = exp_result.priority_terms + exp_result.expanded_terms
            unique_terms = list(dict.fromkeys(terms))
            
            if unique_terms:
                # [수정사항] 확장된 키워드를 원본 user_rules.json(DB) 파일에도 자동 반영!
                updated = False
                for term in unique_terms:
                    if term not in g.related_keywords:
                        g.related_keywords.append(term)
                        updated = True
                
                if updated:
                    # 동일한 goal_id로 덮어쓰며 저장(_save) 수행
                    rule_store.add_goal(g)
                    
                tags = " ".join([f"`{t}`" for t in unique_terms[:15]])
                st.markdown(f"🤖 **AI 매칭 키워드:** {tags}" + ("..." if len(unique_terms) > 15 else ""))
            else:
                st.caption("🤖 AI 쿼리 확장 진행 중 (또는 실패)...")
        except Exception as e:
            st.caption(f"🤖 **AI 쿼리 확장 에러:** {e}")
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 다시 생성", key=f"regen_{g.goal_id}", use_container_width=True):
                # 기존 캐시 삭제 및 키워드 초기화 후 새로고침하여 AI 강제 재호출
                cache_file = Path(f".cache/expansions/{g.goal_id}.json")
                if cache_file.exists():
                    cache_file.unlink()
                g.related_keywords = []
                rule_store.add_goal(g)
                st.rerun()
                
        with col_btn2:
            if st.button("🗑️ 삭제", key=f"del_{g.goal_id}", type="primary", use_container_width=True):
                rule_store.remove_goal(g.goal_id)
                st.rerun()

st.sidebar.subheader("새 목표 추가")
with st.sidebar.form("add_goal_form"):
    st.info("💡 관련된 앱, 도메인, 키워드는 AI가 활동 로그를 보고 알아서 추론합니다. 목표 이름만 편하게 적어주세요!")
    new_title = st.text_input("목표 이름 (예: 파이썬 알고리즘 공부)")
    
    if st.form_submit_button("목표 추가"):
        if new_title.strip():
            auto_id = f"G-{int(time.time())}"
            g = ResearchGoal(
                goal_id=auto_id,
                user_id="u1",
                title=new_title.strip(),
                related_domains=[],
                related_keywords=[]
            )
            rule_store.add_goal(g)
            st.rerun()

st.sidebar.divider()
# ------------------------------

@st.cache_data(ttl=600) # 10분 캐싱: 무거운 API 통신/세션화 작업만 캐싱
def fetch_and_build_sessions():
    raw_events, afk_events = fetch_recent_events(hours=24.0)
    if not raw_events:
        return [], []

    # 전처리 파이프라인
    privacy_filter = PrivacyFilter(Path(".cache/privacy_blacklist.json"))
    safe_events, safe_afk_events = privacy_filter.filter_events_and_afk(raw_events, afk_events)
    
    sessions = build_sessions(safe_events, safe_afk_events)
    sessions = [enrich_session(s) for s in sessions]
    return sessions, safe_afk_events

with st.spinner("ActivityWatch 서버에서 데이터를 가져와 분석 중입니다..."):
    sessions, safe_afk_events = fetch_and_build_sessions()

if not sessions:
    st.error("지난 24시간 동안 수집된 데이터가 없습니다. ActivityWatch 서버가 켜져 있는지 확인해주세요.")
    st.stop()

# 통계 및 목표 매칭 계산 (캐싱 제외 - 목표 변경 즉시 반영)
total_min = sum(s.total_duration_sec for s in sessions) / 60.0

matcher = GoalSessionMatcher(rule_store)
results, unmatched = matcher.match_sessions(current_goals, sessions, interactive=False)

# 통계 추출
goal_stats = {}
matched_min = 0
for goal in current_goals:
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

# 4. 세부 세션 기록 (어떤 로그들이 시간에 합산되었는지)
st.subheader("📝 목표별 상세 합산 로그")
st.markdown("각 목표의 달성 시간에 실질적으로 기여한 활동 내역들입니다.")

for goal in current_goals:
    matched = results[goal.goal_id]
    if not matched:
        continue
        
    with st.expander(f"📌 [{goal.title}] 에 합산된 기록들 ({len(matched)}개 세션)"):
        # 표 형식으로 깔끔하게 정리하기 위해 리스트 생성
        log_data = []
        for m in reversed(matched):
            s = m.session
            
            # 날짜 처리를 위해 문자열이면 datetime 객체로 변환
            try:
                st_dt = s.start_time if isinstance(s.start_time, datetime) else datetime.fromisoformat(s.start_time.replace('Z', '+00:00'))
                et_dt = s.end_time if isinstance(s.end_time, datetime) else datetime.fromisoformat(s.end_time.replace('Z', '+00:00'))
                
                # 날짜가 같으면 뒤에는 시간만, 다르면 뒤에도 날짜 표기
                start_str = st_dt.strftime('%m-%d %H:%M')
                if st_dt.date() == et_dt.date():
                    end_str = et_dt.strftime('%H:%M')
                else:
                    end_str = et_dt.strftime('%m-%d %H:%M')
                
                time_str = f"{start_str} ~ {end_str}"
            except Exception:
                time_str = f"{s.start_time} ~ {s.end_time}"
                
            log_data.append({
                "시간": time_str,
                "소요 시간": f"{s.total_duration_sec/60:.1f} 분",
                "주요 앱": s.primary_app,
                "주요 활동 내역": s.summary_text
            })
        st.dataframe(pd.DataFrame(log_data), use_container_width=True)

st.divider()

# 5. 분류되지 않은 기타/휴식 활동들
st.subheader("🗑️ 기타/휴식 (매칭 안 된 로그)")
st.markdown("어떤 목표와도 연관점을 찾지 못한 활동들입니다. 목표 이름이나 수행 활동에 문제가 있다면 확인해보세요.")

with st.expander(f"📌 분류되지 않은 기록들 ({len(unmatched)}개 세션)"):
    log_data = []
    for s in reversed(unmatched):
        # 문자열을 datetime으로 안전하게 변환
        try:
            st_dt = s.start_time if isinstance(s.start_time, datetime) else datetime.fromisoformat(s.start_time.replace('Z', '+00:00'))
            et_dt = s.end_time if isinstance(s.end_time, datetime) else datetime.fromisoformat(s.end_time.replace('Z', '+00:00'))
            
            start_str = st_dt.strftime('%m-%d %H:%M')
            if st_dt.date() == et_dt.date():
                end_str = et_dt.strftime('%H:%M')
            else:
                end_str = et_dt.strftime('%m-%d %H:%M')
                
            time_str = f"{start_str} ~ {end_str}"
        except Exception:
            time_str = f"{s.start_time} ~ {s.end_time}"
            
        log_data.append({
            "시간": time_str,
            "소요 시간": f"{s.total_duration_sec/60:.1f} 분",
            "주요 앱": s.primary_app,
            "주요 활동 내역": s.summary_text
        })
    if log_data:
        st.dataframe(pd.DataFrame(log_data), use_container_width=True)
    else:
        st.info("매칭되지 않은 로그가 없습니다. 모든 활동이 목표에 잘 분류되었습니다!")
