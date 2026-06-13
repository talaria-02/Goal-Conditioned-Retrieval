try:
    from plyer import notification
except ImportError:
    notification = None

def show_toast_notification(title: str, msg: str, duration: int = 10):
    """Windows 우측 하단에 알림을 띄웁니다 (plyer 사용)."""
    if notification is None:
        print(f"\n[알림 출력 - plyer 미설치]")
        print(f"제목: {title}\n내용: {msg}")
        return
        
    try:
        notification.notify(
            title=title,
            message=msg,
            app_name="AI 행동 코치",
            timeout=duration
        )
    except Exception as e:
        print(f"알림 표시 실패: {e}\n내용: {msg}")
