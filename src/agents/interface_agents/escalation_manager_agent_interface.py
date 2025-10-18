from abc import ABC, abstractmethod
from typing import Dict, Any


class AbstractEscalationManagerAgent(ABC):
    """Абстрактный класс для управления эскалацией"""

    @abstractmethod
    async def escalate_to_human(self, reason: str, context: Dict[str, Any]) -> str:
        """Инициирует эскалацию к человеку-оператору"""
