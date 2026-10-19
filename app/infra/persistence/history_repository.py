from app.shared.clients.mongo_history_utils import (
    clear_history,
    get_recent_messages,
    list_sessions,
    save_chat_message,
    update_message_item_names,
)



class HistoryRepository:
    def list_recent(self,session_id:str,limit:int = 10) -> list[dict]:
        return get_recent_messages(session_id,limit)

    def list_sessions(self, limit: int = 30) -> list[dict]:
        """最近活跃的会话清单（会话 id + 首条用户提问作为标题）。"""
        return list_sessions(limit)


    def save_message(
            self,
            *,
            session_id:str,
            role:str,
            text:str,
            rewritten_query:str = '',
            item_names:list[str] | None = None,
            image_urls:list[str] | None = None,
            message_id:str | None = None,
    ) -> str:
        return save_chat_message(
            session_id = session_id,
            role=role,
            text=text,
            rewritten_query=rewritten_query,
            item_names=item_names,
            image_urls=image_urls,
            message_id=message_id,
        )

    def clear_session(self, session_id: str) -> int:
        return clear_history(session_id)

    def update_item_names(self, ids: list[str], item_names: list[str]) -> int:
        return update_message_item_names(ids, item_names)

history_repository = HistoryRepository()

