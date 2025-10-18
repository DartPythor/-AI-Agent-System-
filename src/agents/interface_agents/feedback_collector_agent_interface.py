from abc import ABC, abstractmethod


class AbstractFeedbackCollectorAgent(ABC):
    """Абстрактный класс для сбора обратной связи"""

    @abstractmethod
    async def collect_feedback(self, user_id: str, conversation_id: str, rating: int, comments: str) -> None:
        """Собирает и сохраняет обратную связь от пользователя"""
