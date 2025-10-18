from abc import ABC, abstractmethod


class AbstractDeEscalationAgent(ABC):
    """Абстрактный класс для де-эскалации конфликтных ситуаций"""

    @abstractmethod
    async def deescalate_conversation(self, toxic_text: str) -> str:
        """Преобразует токсичный запрос в нейтральный и конструктивный"""
