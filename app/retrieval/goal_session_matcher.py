from app.schemas import ResearchGoal, ActivitySession, MatchedSession
from app.retrieval.user_rule_store import UserRuleStore
from app.retrieval.query_understanding import build_query
from app.retrieval.query_expansion import expand_goal_query

class GoalSessionMatcher:
    def __init__(self, rule_store: UserRuleStore):
        self.rule_store = rule_store

    def match_sessions(
        self, goals: list[ResearchGoal], sessions: list[ActivitySession], interactive=True
    ) -> tuple[dict[str, list[MatchedSession]], list[ActivitySession]]:
        """
        주어진 세션들을 목표와 매칭합니다. 
        매칭되지 않은 중요한 세션은 터미널을 통해 사용자에게 인터랙티브하게 물어봅니다.
        """
        results = {g.goal_id: [] for g in goals}
        unmatched = []
        valid_goal_ids = {g.goal_id for g in goals}
        
        # [NEW] 목표별로 AI Query Expansion (키워드 자동 증강) 캐싱
        expanded_vocab = {}
        for goal in goals:
            try:
                q_obj = build_query(goal)
                # Gemini를 이용해 키워드 40~50개 자동 추출 (캐시된 경우 즉시 반환)
                exp = expand_goal_query(goal, q_obj, use_cache=True)
                # 세 가지 부류의 연관어들을 하나의 리스트로 합침
                all_terms = exp.priority_terms + exp.expanded_terms + exp.related_terms
                expanded_vocab[goal.goal_id] = [t.lower() for t in all_terms if len(t) > 1]
            except Exception as e:
                print(f"[{goal.title}] 키워드 자동 확장 실패: {e}")
                expanded_vocab[goal.goal_id] = []

        # 1. 룰 및 지능형 키워드 기반 매칭 시도
        for session in sessions:
            matched_goal_id, reason = self._try_match(session, goals, expanded_vocab)
            
            # 매칭된 ID가 현재 유효한 목표 리스트에 존재하는지 검사 (삭제된 목표의 잔존 룰 방어)
            if matched_goal_id and matched_goal_id in valid_goal_ids:
                results[matched_goal_id].append(
                    MatchedSession(session=session, score=1.0, match_reason=reason)
                )
            else:
                unmatched.append(session)

        # 2. 매칭 실패한 세션에 대해 Interactive Learning
        if interactive and unmatched:
            self._interactive_learn(unmatched, goals, results)

        return results, unmatched

    def _try_match(self, session: ActivitySession, goals: list[ResearchGoal], expanded_vocab: dict) -> tuple[str | None, str]:
        # 1) 사용자가 이전에 등록한 Custom Rule 확인 (가장 우선순위 높음)
        if session.primary_domain:
            goal_id = self.rule_store.get_goal_for_domain(session.primary_domain)
            if goal_id: return goal_id, "User Custom Domain Rule"
        
        goal_id = self.rule_store.get_goal_for_app(session.primary_app)
        if goal_id: return goal_id, "User Custom App Rule"

        # 세션의 텍스트들을 하나의 덩어리로 합쳐서 검색 풀 생성
        session_text_pool = f"{session.summary_text} {session.primary_domain} {session.primary_app} {' '.join(session.keywords)}".lower()
        session_text_nospaces = session_text_pool.replace(" ", "")

        # 2) AI가 자동 증강한 키워드(Query Expansion) 매칭
        for goal in goals:
            # 먼저 기존 하드코딩된 앱/도메인 매칭 (호환성)
            if session.primary_domain in goal.related_domains:
                return goal.goal_id, f"Hardcoded Domain ({session.primary_domain})"
            if session.primary_app in goal.related_apps:
                return goal.goal_id, f"Hardcoded App ({session.primary_app})"
            
            # AI 확장 키워드 + 목표 제목 자체의 단어들도 검사
            ai_terms = expanded_vocab.get(goal.goal_id, [])
            title_terms = [t.lower() for t in goal.title.split() if len(t) > 1]
            all_search_terms = set(ai_terms + title_terms)
            
            for term in all_search_terms:
                term_nospaces = term.replace(" ", "")
                # 일반 포함 여부 및 띄어쓰기 무시 포함 여부 검사
                if term in session_text_pool or (len(term_nospaces) > 1 and term_nospaces in session_text_nospaces):
                    return goal.goal_id, f"Keyword Match ({term})"

        return None, ""

    def _interactive_learn(self, unmatched: list[ActivitySession], goals: list[ResearchGoal], results: dict):
        print("\n" + "="*60)
        print("🧠 [Interactive Learning] 분류되지 않은 활동 패턴이 발견되었습니다.")
        print("="*60)
        
        for session in unmatched:
            # 1분 미만의 너무 짧은 세션이나 단순 빈 브라우저(newtab)는 무시
            if session.total_duration_sec < 60 or session.summary_text == "newtab 에서 활동":
                continue
                
            print(f"\n❓ 알 수 없는 활동 발견:")
            print(f"  - 앱/도메인: [{session.primary_app}] {session.primary_domain}")
            print(f"  - 활동 시간: {session.total_duration_sec/60:.1f} 분")
            print(f"  - 상세 요약: {session.summary_text}")
            
            print("\n이 활동은 어떤 목표와 관련이 있나요?")
            for idx, goal in enumerate(goals):
                print(f"  [{idx+1}] {goal.title}")
            print(f"  [{len(goals)+1}] 관련 없음 (휴식/기타)")
            
            try:
                choice = input("👉 선택 (번호 입력): ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(goals):
                    target_goal = goals[choice_idx]
                    
                    # Rule Store에 저장
                    if session.primary_domain:
                        self.rule_store.add_domain_rule(session.primary_domain, target_goal.goal_id)
                        print(f"✅ 앞으로 도메인 '{session.primary_domain}'은(는) '{target_goal.title}'에 자동으로 매칭됩니다.")
                    else:
                        self.rule_store.add_app_rule(session.primary_app, target_goal.goal_id)
                        print(f"✅ 앞으로 앱 '{session.primary_app}'은(는) '{target_goal.title}'에 자동으로 매칭됩니다.")
                        
                    results[target_goal.goal_id].append(
                        MatchedSession(session=session, score=0.8, match_reason="Interactive Learned Rule")
                    )
                else:
                    print("⏭️ 관련 없는 활동으로 패스합니다.")
            except ValueError:
                print("⏭️ 입력 오류로 패스합니다.")
