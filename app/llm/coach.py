import os
from google import genai

class RealtimeCoach:
    def __init__(self, api_key: str = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        # 최신 google-genai 클라이언트 초기화
        self.client = genai.Client(api_key=key)
        
        # 환경변수에서 모델명을 가져오고, 없으면 기본값(gemini-2.5-flash) 사용
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def generate_feedback(self, goal_stats: dict, total_duration_min: float,
                          health_stats: dict = None, past_context: str = "") -> str:
        """통계(딥워크, 헬스케어, 과거 RAG 포함)를 바탕으로 코칭 텍스트를 생성합니다."""
        
        stats_text = f"총 수집된 유효 활동 시간: {total_duration_min:.1f} 분\n"
        for goal_title, stats in goal_stats.items():
            t_min = stats.get("total_min", 0)
            c_min = stats.get("max_continuous_min", 0)
            count = stats.get("session_count", 0)
            stats_text += f" - [{goal_title}]: 총 {t_min:.1f}분 (최대 연속 집중: {c_min:.1f}분, 흐름 끊김: {count}회)\n"

        if health_stats:
            t_afk = health_stats.get("total_afk_min", 0)
            m_afk = health_stats.get("max_single_afk_min", 0)
            stats_text += f"\n[헬스케어 통계]\n - 총 휴식/자리비움 시간: {t_afk:.1f}분 (가장 길게 쉰 시간: {m_afk:.1f}분)\n"

        # 과거 유사 세션 RAG 컨텍스트
        past_section = ""
        if past_context:
            past_section = f"""
[과거 유사 활동 기록 (RAG 검색 결과)]
{past_context}
"""

        prompt = f"""당신은 사용자의 생산성과 웰니스(Wellness)를 관리하는 예리한 'AI 행동 코치'입니다.
아래는 사용자의 최근 시간(예: 1시간) 동안의 활동 및 헬스케어 통계입니다.

[활동 통계]
{stats_text}
{past_section}
위 통계를 분석하여, 윈도우 푸시 알림으로 띄워줄 2~3문장의 짧은 피드백을 작성하세요.
- [딥워크 평가] 총 시간은 길지만 '최대 연속 집중 시간'이 짧고 '흐름 끊김'이 많다면 주의가 산만함을 지적하세요. 반대로 연속 집중 시간이 길다면 극찬하세요.
- [헬스케어 평가] 활동 시간이 긴데 '가장 길게 쉰 시간'이 3분 미만이라면 뇌와 눈 건강을 위해 잠시 일어서서 휴식할 것을 강하게 권장하세요.
- [과거 비교] 만약 '과거 유사 활동 기록'이 제공되었다면, 과거와 현재의 집중 패턴을 비교하여 개선점이나 퇴보를 짚어주세요. 예를 들어 "지난번에는 더 집중했었는데 오늘은..." 식으로.
- 불필요한 인사말 없이 바로 본론만 말하세요. 스마트폰 알람처럼 친근하지만 전문가적인 어투를 사용하세요.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"피드백 생성 실패 ({self.model_name}): {e}"
