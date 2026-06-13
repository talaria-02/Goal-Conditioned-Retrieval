"""Enrich sessions with meaning derived from URLs and titles."""
import re
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from app.schemas import ActivitySession
from app.llm.llm_client import get_llm_client

LLM_SUMMARY_CACHE_PATH = Path(".cache/llm_session_summaries.json")

def _get_llm_summary(titles: list[str], urls: list[str], fallback_summary: str) -> str:
    """창 제목 목록을 바탕으로 LLM에게 현재 작업 맥락의 요약을 요청합니다."""
    valid_titles = [t for t in titles if t and t != "unknown"]
    if not valid_titles:
        return fallback_summary
        
    # 캐시 키 생성 (타이틀 묶음으로 해시)
    key_str = "|".join(sorted(set(valid_titles)))
    cache_key = hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    # 캐시 로드
    cache = {}
    if LLM_SUMMARY_CACHE_PATH.exists():
        try:
            with open(LLM_SUMMARY_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass
            
    if cache_key in cache:
        return cache[cache_key]
        
    # 캐시 미스 -> LLM 호출
    llm = get_llm_client(mock=False)
    
    prompt = f"""사용자의 최근 컴퓨터 활동 기록(창 제목 목록)입니다:
{chr(10).join(f'- {t}' for t in valid_titles)}

이 기록들을 바탕으로 사용자가 현재 어떤 맥락(Context)의 작업을 하고 있는지 1~2문장으로 요약해 주세요.
프로젝트 이름, 사용 중인 파일명/기술, 구체적인 작업 내용(예: 코딩, 문서 읽기, 자료 조사, 웹서핑 등)을 포함하여 직관적으로 작성해 주세요.
불필요한 인사말 없이 요약된 결과만 바로 출력하세요."""

    try:
        summary = llm.generate(prompt).strip()
        # 캐시 저장
        cache[cache_key] = summary
        LLM_SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LLM_SUMMARY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        return summary
    except Exception as e:
        print(f"LLM Summary Error: {e}")
        return fallback_summary


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
            
            # 1. 파일명 추출
            match_file = re.search(r"([a-zA-Z0-9_]+\.(?:py|js|ts|html|css|json|md))", title)
            if match_file:
                keywords.add(match_file.group(1))
                
            # 2. 프로젝트명 추출 (Phase A 로직)
            # 패턴: "파일명 - 프로젝트명 - IDE이름" (예: "privacy_filter.py - 목표기반 RAG - Antigravity IDE")
            parts = [p.strip() for p in title.split("-")]
            if len(parts) >= 3:
                # 대개 뒤에서 두 번째 부분이 프로젝트/폴더명입니다.
                project_name = parts[-2]
                if 2 <= len(project_name) <= 30:
                    keywords.add(project_name)
                
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
        
    fallback_summary = " — ".join(summary_parts)
    
    # 5. LLM을 이용한 문맥 기반 강력한 요약 텍스트 생성 (캐싱 적용)
    session.summary_text = _get_llm_summary(session.titles, session.urls, fallback_summary)
    
    return session
