import json
from pathlib import Path

from app.schemas import ResearchGoal

class UserRuleStore:
    """사용자가 동적으로 추가한 목표(Goal)와 도메인/앱 룰을 저장하고 관리하는 클래스."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.rules = self._load()

    def _load(self) -> dict:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "goals" not in data:
                        data["goals"] = []
                    return data
            except json.JSONDecodeError:
                pass
        return {"domains": {}, "apps": {}, "goals": []}

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

    # --- Goal(목표) 동적 관리 ---
    
    def get_all_goals(self) -> list[ResearchGoal]:
        """저장된 목표 목록을 ResearchGoal 객체 리스트로 반환합니다."""
        goal_list = []
        for g in self.rules.get("goals", []):
            goal_list.append(ResearchGoal(
                goal_id=g.get("goal_id", ""),
                user_id=g.get("user_id", "u1"),
                title=g.get("title", ""),
                description=g.get("description", ""),
                related_domains=g.get("related_domains", []),
                related_apps=g.get("related_apps", []),
                related_keywords=g.get("related_keywords", [])
            ))
        return goal_list

    def add_goal(self, goal: ResearchGoal):
        """새로운 목표를 추가합니다. 이미 같은 ID가 있으면 덮어씁니다."""
        goals_data = self.rules.setdefault("goals", [])
        
        # 중복 방지: 같은 ID가 있으면 제거
        goals_data = [g for g in goals_data if g.get("goal_id") != goal.goal_id]
        
        goals_data.append({
            "goal_id": goal.goal_id,
            "user_id": goal.user_id,
            "title": goal.title,
            "description": goal.description,
            "related_domains": goal.related_domains,
            "related_apps": goal.related_apps,
            "related_keywords": goal.related_keywords
        })
        self.rules["goals"] = goals_data
        self._save()

    def remove_goal(self, goal_id: str):
        """목표를 삭제하고, 해당 목표와 연관된 도메인/앱 매칭 룰도 정리합니다."""
        # 1. 목표 삭제
        goals_data = self.rules.setdefault("goals", [])
        self.rules["goals"] = [g for g in goals_data if g.get("goal_id") != goal_id]
        
        # 2. 관련 매칭 룰(domains, apps) 클린업
        domains = self.rules.setdefault("domains", {})
        apps = self.rules.setdefault("apps", {})
        
        keys_to_delete = [domain for domain, gid in domains.items() if gid == goal_id]
        for key in keys_to_delete:
            del domains[key]
            
        keys_to_delete = [app for app, gid in apps.items() if gid == goal_id]
        for key in keys_to_delete:
            del apps[key]
            
        self._save()
