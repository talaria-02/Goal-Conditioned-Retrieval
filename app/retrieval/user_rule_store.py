import json
from pathlib import Path

class UserRuleStore:
    """사용자가 동적으로 추가한 도메인/앱 룰을 저장하고 관리하는 클래스."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.rules = self._load()

    def _load(self) -> dict:
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"domains": {}, "apps": {}}

    def _save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)

    def add_domain_rule(self, domain: str, goal_id: str):
        if not domain: return
        self.rules["domains"][domain] = goal_id
        self._save()

    def add_app_rule(self, app: str, goal_id: str):
        if not app: return
        self.rules["apps"][app] = goal_id
        self._save()

    def get_goal_for_domain(self, domain: str) -> str | None:
        if not domain: return None
        return self.rules["domains"].get(domain)

    def get_goal_for_app(self, app: str) -> str | None:
        if not app: return None
        return self.rules["apps"].get(app)
