import json
from pathlib import Path
from typing import List, Tuple
from datetime import timedelta
from app.preprocessing.aw_parser import RawEvent, AfkEvent

class PrivacyFilter:
    """사용자의 민감한 기록을 지우는 대신, 안전한 카테고리 이름으로 덮어씌워(Redact) 프라이버시를 보호합니다."""
    
    # 카테고리 템플릿 (초기엔 비워두고 사용자가 직접 json을 수정하여 추가)
    DEFAULT_CATEGORIES = {
        "SOCIAL_MESSENGER": {
            "apps": [],
            "domains": [],
            "keywords": []
        },
        "ENTERTAINMENT_MEDIA": {
            "apps": [],
            "domains": [],
            "keywords": []
        },
        "FINANCE_PERSONAL": {
            "apps": [],
            "domains": [],
            "keywords": []
        },
        "SHOPPING": {
            "apps": [],
            "domains": [],
            "keywords": []
        }
    }

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.categories = self._load()

    def _load(self) -> dict:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 구버전 포맷 감지 ("apps"가 최상위 키로 있으면 구버전)
                    if "apps" in data and isinstance(data["apps"], list):
                        print("구버전 프라이버시 블랙리스트 포맷이 감지되었습니다. 새 템플릿으로 초기화합니다.")
                    else:
                        return data
            except json.JSONDecodeError:
                pass # 파일이 깨진 경우 무시
                
        # 파일이 없거나 구버전이면 기본 설정값을 반환하고 즉시 저장
        self._save(self.DEFAULT_CATEGORIES)
        return self.DEFAULT_CATEGORIES.copy()

    def _save(self, data=None):
        target_data = data if data else self.categories
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(target_data, f, ensure_ascii=False, indent=2)

    def get_sensitive_category(self, event: RawEvent) -> str | None:
        """이벤트가 속한 민감 카테고리 이름을 반환합니다. 안전하면 None 반환."""
        for cat_name, rules in self.categories.items():
            # 1. 앱 이름 체크
            for b_app in rules.get("apps", []):
                if b_app.lower() in event.app.lower():
                    return cat_name
            # 2. 도메인 / URL 체크
            if event.url:
                for b_domain in rules.get("domains", []):
                    if b_domain.lower() in event.url.lower():
                        return cat_name
            # 3. 키워드 체크
            if event.title:
                for b_word in rules.get("keywords", []):
                    if b_word.lower() in event.title.lower():
                        return cat_name
        return None

    def filter_events_and_afk(self, events: List[RawEvent], afk_events: List[AfkEvent]) -> Tuple[List[RawEvent], List[AfkEvent]]:
        """
        민감한 이벤트 발견 시, 해당 이벤트를 지우는 대신
        해당 시간대의 모든 이벤트 Title과 URL을 '[PRIVATE: 카테고리명]' 으로 안전하게 덮어씌웁니다.
        """
        # 1. 어떤 시간 구간에, 어떤 민감 카테고리가 떴는지 추출
        sensitive_intervals = [] # (start, end, category)
        for event in events:
            cat = self.get_sensitive_category(event)
            if cat:
                start = event.timestamp
                end = start + timedelta(seconds=event.duration)
                sensitive_intervals.append((start, end, cat))

        if not sensitive_intervals:
            return events, afk_events

        # 2. 겹치는 구간 병합 (Merge Intervals)
        # 겹치는 구간은 먼저 발견된 카테고리로 통일
        sensitive_intervals.sort(key=lambda x: x[0])
        merged = [sensitive_intervals[0]]
        for current_start, current_end, current_cat in sensitive_intervals[1:]:
            last_start, last_end, last_cat = merged[-1]
            if current_start <= last_end: # 겹침
                merged[-1] = (last_start, max(last_end, current_end), last_cat)
            else:
                merged.append((current_start, current_end, current_cat))

        # 3. 해당 시간대와 겹치는지 확인하고 카테고리 반환
        def get_overlap_category(start, end):
            for m_start, m_end, m_cat in merged:
                if max(start, m_start) < min(end, m_end):
                    return m_cat
            return None

        # 4. Safe Events 가공 (지우지 않고 덮어쓰기 Redact)
        safe_events = []
        for e in events:
            e_start = e.timestamp
            e_end = e_start + timedelta(seconds=max(e.duration, 0.001))
            overlap_cat = get_overlap_category(e_start, e_end)
            
            if overlap_cat:
                # 민감 정보 덮어쓰기 (Redaction)
                e.title = f"[PRIVATE: {overlap_cat}]"
                if e.url:
                    e.url = f"[REDACTED_URL]"
                # 앱 이름이 노골적인 경우를 대비해 앱 이름도 덮어씀
                if e.app != "Chrome" and e.app != "Google-chrome":
                    e.app = "PrivateApp"
                    
            safe_events.append(e)

        # 5. AFK는 텍스트 정보가 없으므로 건드릴 필요 없이 그대로 반환
        return safe_events, afk_events
