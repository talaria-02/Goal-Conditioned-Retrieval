"""세션 히스토리 저장/로드 및 ActivitySession → ResearchLog 변환 어댑터.

매시간 수집된 ActivitySession을 JSON 파일로 누적 저장하고,
기존 RAG 파이프라인(DenseRetriever)이 이해할 수 있는 ResearchLog 형태로 변환합니다.
"""
import json
from pathlib import Path
from datetime import datetime
from app.schemas import ActivitySession, ResearchLog


def session_to_research_log(session: ActivitySession) -> ResearchLog:
    """ActivitySession 하나를 기존 RAG의 ResearchLog로 변환합니다."""
    return ResearchLog(
        log_id=session.session_id,
        user_id="u1",
        date=str(session.start_time)[:10],  # "2026-06-13"
        title=session.summary_text,
        content=f"앱: {session.primary_app}, 도메인: {session.primary_domain}, "
                f"시간: {session.total_duration_sec / 60:.1f}분, "
                f"키워드: {', '.join(session.keywords[:5])}",
        activity_type=session.activity_category,
        metadata={
            "duration_sec": session.total_duration_sec,
            "primary_app": session.primary_app,
            "primary_domain": session.primary_domain,
            "keywords": session.keywords,
        },
        timestamp=str(session.start_time),
    )


class SessionHistoryStore:
    """세션 기록을 로컬 JSON 파일에 누적 저장/로드합니다."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_sessions(self, sessions: list[ActivitySession]):
        """새로운 세션들을 기존 히스토리에 추가 저장합니다."""
        existing = self._load_raw()

        for s in sessions:
            entry = {
                "session_id": s.session_id,
                "start_time": str(s.start_time),
                "end_time": str(s.end_time),
                "total_duration_sec": s.total_duration_sec,
                "primary_app": s.primary_app,
                "primary_domain": s.primary_domain,
                "activity_category": s.activity_category,
                "summary_text": s.summary_text,
                "keywords": s.keywords,
                "saved_at": datetime.now().isoformat(),
            }
            # 중복 방지 (같은 session_id가 이미 있으면 스킵)
            if not any(e["session_id"] == s.session_id for e in existing):
                existing.append(entry)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2, default=str)

    def load_as_research_logs(self) -> list[ResearchLog]:
        """저장된 히스토리를 ResearchLog 리스트로 변환하여 반환합니다."""
        raw = self._load_raw()
        logs = []
        for entry in raw:
            logs.append(ResearchLog(
                log_id=entry["session_id"],
                user_id="u1",
                date=entry["start_time"][:10],
                title=entry["summary_text"],
                content=f"앱: {entry['primary_app']}, 도메인: {entry['primary_domain']}, "
                        f"시간: {entry['total_duration_sec'] / 60:.1f}분, "
                        f"키워드: {', '.join(entry.get('keywords', [])[:5])}",
                activity_type=entry.get("activity_category", "unknown"),
                metadata={
                    "duration_sec": entry["total_duration_sec"],
                    "primary_app": entry["primary_app"],
                    "primary_domain": entry["primary_domain"],
                    "keywords": entry.get("keywords", []),
                },
                timestamp=entry["start_time"],
            ))
        return logs

    def _load_raw(self) -> list[dict]:
        if not self.file_path.exists():
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []
