from abc import ABC, abstractmethod
from typing import List


class AbstractResponseRegeneratorAgent(ABC):
    """Абстрактный класс для перегенерации небезопасных ответов"""

    @abstractmethod
    async def regenerate_unsafe_response(self, unsafe_response: str, safety_issues: List[str]) -> str:
        """Перегенерирует ответ, устраняя проблемы безопасности"""
