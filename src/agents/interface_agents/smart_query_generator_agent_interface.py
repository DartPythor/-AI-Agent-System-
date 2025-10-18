from abc import ABC, abstractmethod
from typing import List


class AbstractSmartQueryGeneratorAgent(ABC):
    """Абстрактный класс для генерации уточняющих вопросов"""

    @abstractmethod
    async def generate_clarifying_questions(self, information_gaps: List[str]) -> List[str]:
        """Генерирует уточняющие вопросы для заполнения пробелов"""
