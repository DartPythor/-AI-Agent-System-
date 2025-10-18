from abc import ABC, abstractmethod
from typing import List, Dict


class AbstractInformationGapAnalyzerAgent(ABC):
    """Абстрактный класс для анализа информационных пробелов"""

    @abstractmethod
    async def analyze_gaps(self, user_request: str, current_context: Dict) -> List[str]:
        """Выявляет недостающую информацию для обработки запроса"""
