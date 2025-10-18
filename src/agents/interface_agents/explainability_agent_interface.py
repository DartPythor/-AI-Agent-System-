from abc import ABC, abstractmethod
from typing import Dict, Any


class AbstractExplainabilityAgent(ABC):
    """Абстрактный класс для обеспечения прозрачности решений"""

    @abstractmethod
    async def generate_explanation(self, decision_context: Dict[str, Any]) -> str:
        """Генерирует объяснение принятого решения"""
