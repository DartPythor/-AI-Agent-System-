from abc import ABC, abstractmethod
from typing import List, Dict, Any


class AbstractFailureDetectorAgent(ABC):
    """Абстрактный класс для мониторинга системных сбоев"""

    @abstractmethod
    async def detect_failures(self) -> List[Dict[str, Any]]:
        """Обнаруживает системные сбои и ошибки в реальном времени"""
