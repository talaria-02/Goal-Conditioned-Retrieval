"""Enrich sessions with meaning derived from URLs and titles."""
import re
from collections import defaultdict
from app.schemas import ActivitySession


def enrich_session(session: ActivitySession) -> ActivitySession:
    """
    URL/title 패턴에서 의미(activity_category, summary_text, keywords) 추출.
    """
    keywords = set()
    categories = defaultdict(int)
    summary_parts = []
    
    # 1. Title 기반 키워드 추출 및 카테고리 매핑
    for title in session.titles:
        if not title or title == "unknown":
            continue
        
        # IDE / Coding
        if "Antigravity IDE" in title or "Code" in title or ".py" in title:
            categories["coding"] += 1
            # Extract filename if possible (e.g., "arm_controller.py - Antigravity")
            match = re.search(r"([a-zA-Z0-9_]+\.(?:py|js|ts|html|css|json|md))", title)
            if match:
                keywords.add(match.group(1))
                
        # Gemini / AI
        if "Google Gemini" in title:
            categories["ai_chat"] += 1
            # Extract topic: "최적화 기법 및 회귀 실습 - Google Gemini" -> "최적화 기법 및 회귀 실습"
            topic = title.replace("- Google Gemini", "").strip()
            if topic and topic != "Google Gemini":
                keywords.add(topic)
                
    # 2. URL/Domain 기반 매핑
    for url in session.urls:
        if "github.com" in url:
            categories["coding"] += 1
            # Extract repo name
            parts = url.split("github.com/")
            if len(parts) > 1:
                repo_path = parts[1].split("/")
                if len(repo_path) >= 2:
                    repo_name = f"{repo_path[0]}/{repo_path[1]}"
                    keywords.add(repo_name)
                    
        elif "cyber.gachon.ac.kr" in url:
            categories["university"] += 1
            keywords.add("LMS")
            keywords.add("대학 수업")
            
        elif "google.com/search" in url:
            categories["browsing"] += 1
            match = re.search(r"q=([^&]+)", url)
            if match:
                query = match.group(1).replace("+", " ")
                import urllib.parse
                try:
                    query = urllib.parse.unquote(query)
                    keywords.add(query)
                except Exception:
                    pass

    # 3. Best Category Decision
    if categories:
        best_category = max(categories.items(), key=lambda x: x[1])[0]
    else:
        # Fallback to primary app
        app = session.primary_app.lower()
        if "chrome" in app or "browser" in app:
            best_category = "browsing"
        elif "terminator" in app or "code" in app or "ide" in app:
            best_category = "coding"
        else:
            best_category = "unknown"
            
    session.activity_category = best_category
    
    # 4. Generate Summary Text
    # Convert keywords to list and build summary
    session.keywords = list(keywords)
    
    if best_category == "coding":
        files = [k for k in session.keywords if "." in k or "/" in k]
        if files:
            summary_parts.append(f"개발/코딩: {', '.join(files[:3])} 등 작업")
        else:
            summary_parts.append("개발/코딩 작업 진행")
            
    elif best_category == "ai_chat":
        topics = [k for k in session.keywords if "." not in k and "/" not in k]
        if topics:
            summary_parts.append(f"AI 활용/질문: {', '.join(topics[:3])}")
        else:
            summary_parts.append("AI 어시스턴트 활용")
            
    elif best_category == "university":
        summary_parts.append("대학 강의 수강 및 LMS 활동")
        
    elif session.primary_domain:
        summary_parts.append(f"{session.primary_domain} 에서 활동")
    else:
        summary_parts.append(f"{session.primary_app} 애플리케이션 사용")
        
    session.summary_text = " — ".join(summary_parts)
    
    return session
