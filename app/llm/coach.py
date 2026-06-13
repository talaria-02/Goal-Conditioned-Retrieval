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

        prompt = f"""당신은 사용자의 행동 로그를 분석하여 팩트 위주의 냉정하고 객관적인 피드백을 제공하는 데이터 분석가입니다.
아래는 사용자의 최근 활동 및 헬스케어 통계입니다.

[활동 통계]
{stats_text}
{past_section}
위 통계를 분석하여, 윈도우 푸시 알림으로 띄워줄 짧은 피드백(2~3문장)을 작성하세요.

[제약 조건] 
1. 과장된 칭찬, 감탄사, 심리학적 분석, 오지랖 넓은 훈수를 절대 금지합니다.
2. 철저하게 데이터(집중 시간, 끊김 횟수, 휴식 시간)에 기반하여 건조하고 직관적인 평가만 작성하세요.

[평가 지침]
- [딥워크] '총 시간', '최대 연속 집중 시간', '흐름 끊김 횟수'를 수치 그대로 짚어주며 현재 집중도의 효율을 냉정하게 브리핑하세요.
- [휴식] '가장 길게 쉰 시간'이 3분 미만이라면 "장시간 연속 작업 중. 최소 3분의 눈/신체 휴식이 권장됨." 정도로만 건조하게 통보하세요.
- [과거 비교] 과거 기록이 제공되었다면 이전 수치와 비교하여 증감만 짧게 보고하세요.
- 불필요한 인사말 없이 바로 본론만 출력하세요.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"피드백 생성 실패 ({self.model_name}): {e}"
