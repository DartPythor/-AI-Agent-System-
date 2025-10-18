from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import SafetyCheckResult


class AbstractResponseSafetyCheckAgent(ABC):
    """Абстрактный класс для проверки безопасности ответов"""

    @abstractmethod
    async def check_response_safety(self, response: str) -> SafetyCheckResult:
        """Проверяет сгенерированный ответ на соответствие политикам безопасности"""
