# Goal-Conditioned Retrieval

ActivityWatch의 실시간 행동 로그를 수집·분석하여, 사용자의 **목표 달성도**를 추적하고
**AI 코치가 맞춤형 피드백**을 제공하는 Goal-Conditioned Evidence Retrieval 시스템.

---

## 시스템 개요

본 시스템은 크게 두 가지 모듈로 구성됩니다.

1. **실시간 AI 코치 파이프라인** — ActivityWatch REST API로부터 실시간 행동 데이터를 수집하고, 프라이버시 보호(Redaction) → 세션화 → 목표 매칭 → 과거 유사 세션 RAG 검색 → LLM 코칭 피드백 생성 → 윈도우 알림까지 자동으로 수행합니다.
2. **Goal-Conditioned Evidence Retrieval** — Gemini Embedding 기반 Dense Retrieval + Lexical Reranker를 활용하여, 사용자의 활동 로그 중 특정 목표와 관련된 증거를 정밀하게 검색합니다.

---

## 실시간 AI 코치 파이프라인

```
ActivityWatch (localhost:5600)
 │  REST API (최근 N시간)
 ▼
[Phase 1] Data Fetching
 │  Window / Web / AFK 이벤트 수집 → RawEvent, AfkEvent 변환
 ▼
[Phase 2] Privacy Filter (Redaction)
 │  민감 이벤트 제목/URL을 [PRIVATE: 카테고리명]으로 익명화
 │  카테고리: SOCIAL_MESSENGER, ENTERTAINMENT_MEDIA, FINANCE_PERSONAL, SHOPPING
 │  사용자가 .cache/privacy_blacklist.json을 직접 편집하여 룰 커스텀
 ▼
[Phase 3] Session Builder + Enricher
 │  60초 이내 동일 앱/도메인 이벤트 병합 → ActivitySession
 │  AFK 시간 감산, 메타데이터 추출
 ▼
[Phase 4] Goal-Session Matcher
 │  사용자 정의 목표(ResearchGoal)에 세션을 분류
 │  딥워크(최대 연속 집중 시간) + 헬스케어(AFK 패턴) 통계 추출
 ▼
[Phase 5] Behavioral RAG (과거 세션 검색)
 │  세션 히스토리를 ResearchLog로 변환 → DenseRetriever 인덱싱
 │  현재 목표와 유사한 과거 세션을 Gemini Embedding으로 검색
 │  과거 패턴을 LLM 프롬프트에 주입 → "과거의 나"와 비교하는 코칭
 ▼
[Phase 6] LLM Coach (Gemini)
 │  활동 통계 + 딥워크 + 헬스케어 + RAG 과거 비교 → 2~3문장 피드백 생성
 ▼
[Phase 7] Windows Push Notification (plyer)
```

### 주요 특징

| 기능 | 설명 |
|---|---|
| **Privacy Redaction** | 이벤트를 삭제하지 않고 `[PRIVATE: 카테고리명]`으로 덮어씌워 시간 통계 정확성 유지 |
| **딥워크 분석** | 총 작업 시간뿐 아니라 "끊김 없이 연속 집중한 최대 시간"을 측정 |
| **헬스케어 모니터링** | AFK 패턴을 분석하여 장시간 무휴식 작업 시 휴식 권장 |
| **Behavioral RAG** | 기존 DenseRetriever를 활용, 과거 유사 세션을 검색하여 패턴 비교 코칭 |
| **사용자 주도 커스텀** | 프라이버시 블랙리스트, 목표, 매칭 룰 모두 사용자가 직접 편집 가능 |

---

## 동적 텍스트 매칭 엔진 (Goal-Session Matcher)

과거 복잡한 Dense/Lexical 혼합 수식을 벗어나, **로컬 환경에서의 실시간 속도와 정확도**를 위해 "지능형 키워드 매칭 + 사용자 맞춤형 룰 학습" 구조로 개편되었습니다.

```text
[입력] 목표(Goal) + 현재 활동 로그(Session)
  │
  ▼
[AI 쿼리 확장] Gemini API를 이용한 하이브리드 키워드 생성
  │  - 영문 원시 로그(앱/도메인) 50%
  │  - 한글 행동 요약문(명사구) 50%
  ▼
[매칭 판별]
  1순위: 사용자가 직접 지정한 도메인/앱 규칙 (User Custom Rules)
  2순위: AI가 확장한 하이브리드 키워드가 Session 텍스트 풀에 포함되는지 검사
  ▼
[인터랙티브 학습]
  매칭되지 않은 중요한 활동은 터미널에서 사용자에게 직접 물어보고,
  새로운 도메인/앱 매칭 룰로 영구 저장 (Personalized Learning)
```

---

## Project Structure

```
app/
  config.py                           # 전역 설정
  schemas.py                          # ResearchGoal, ResearchLog, ActivitySession, ...

  preprocessing/                      # [신규] 실시간 데이터 전처리
    api_fetcher.py                    #   ActivityWatch REST API 연동
    aw_parser.py                      #   RawEvent, AfkEvent 데이터 구조
    privacy_filter.py                 #   카테고리 기반 Redaction 필터
    session_builder.py                #   시간 기반 세션 병합
    session_enricher.py               #   세션 메타데이터 추출

  retrieval/
    goal_session_matcher.py           #   [핵심] 동적 키워드 기반 목표-세션 매칭
    query_expansion.py                #   [핵심] AI 하이브리드 어휘 확장
    session_history.py                #   세션 히스토리 저장 및 관리
    user_rule_store.py                #   사용자 커스텀 매칭 룰 (학습 데이터)

  llm/
    llm_client.py                     #   LLM 인터페이스 연동
    coach.py                          #   실시간 AI 코칭 피드백 생성
    analysis.py                       #   활동 통계 요약

run_realtime_coach.py                 # [실행] 실시간 AI 코치 (CLI 백그라운드)
run_dashboard.py                      # [실행] Streamlit 분석 대시보드 (Web UI)

scripts/
  test_fetcher.py                     #   API 연결 테스트
  test_matcher.py                     #   목표 매칭 테스트
  test_preprocessing.py               #   전처리 파이프라인 테스트

.cache/
  privacy_blacklist.json              #   프라이버시 필터 설정 (사용자 편집)
  session_history.json                #   과거 세션 히스토리 (RAG용)
  user_rules.json                     #   사용자 커스텀 매칭 룰
  embeddings/                         #   임베딩 디스크 캐시
```

---

## Setup

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 의존성 설치
pip install -r requirements.txt
pip install google-genai python-dotenv plyer streamlit pandas
```

`.env` 파일 생성:
```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

**필수 조건:** [ActivityWatch](https://activitywatch.net/)가 로컬에서 실행 중이어야 합니다 (기본 포트: 5600).

---

## Quick Start

### 실시간 AI 코치 (CLI)

```bash
# 최근 1시간 분석 후 피드백 알림
python run_realtime_coach.py

# 최근 3시간 분석
python run_realtime_coach.py --hours 3.0
```

### 대시보드 (Web UI)

```bash
python -m streamlit run run_dashboard.py
```

### Evidence Retrieval (Stage 1)

```bash
python scripts/run_stage1.py --goal_id G-U0003-05 --baseline ours
```

---

## Embedding Providers

| Provider | 조건 | 특징 |
|---|---|---|
| `GeminiEmbeddingProvider` | `GEMINI_API_KEY` 설정 | `gemini-embedding-001`, 3072-dim, 한국어 지원, 비대칭(doc/query 분리) |
| `MockEmbeddingProvider` | API 키 없음 (fallback) | SHA256 기반 결정적 벡터 (semantic 의미 없음) |

---

## Limitations

- **ActivityWatch 의존**: AW 서버가 로컬에서 실행 중이어야 실시간 데이터 수집 가능
- **Synthetic data only**: Evidence Retrieval 평가는 합성 데이터 기반
- **Single embedding space**: 목표(추상)와 로그(구체) 간 의미적 층위 차이
- **Privacy filter 초기 상태**: 블랙리스트가 비어있으므로 사용자가 직접 채워야 함
