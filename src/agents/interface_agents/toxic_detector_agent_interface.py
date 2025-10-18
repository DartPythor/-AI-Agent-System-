from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import SafetyCheckResult


class AbstractToxicityDetectorAgent(ABC):
    """Абстрактный класс для детекции токсичности"""

    @abstractmethod
    async def detect_toxicity(self, text: str) -> SafetyCheckResult:
        """Анализирует текст на наличие токсичности и оскорбительного контента"""
