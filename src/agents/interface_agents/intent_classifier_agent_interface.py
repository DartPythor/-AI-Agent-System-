from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import Response


class AbstractIntentClassifierAgent(ABC):
    """Абстрактный класс для классификации обращения"""

    @abstractmethod
    async def get_classification(self, data: str) -> Response:
        """Возвращает классификацию запроса и уверенность в возможности дать ответ"""
