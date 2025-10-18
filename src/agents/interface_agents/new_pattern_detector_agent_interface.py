from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import UserRequest
from typing import List, Dict, Any


class AbstractNewPatternDetectorAgent(ABC):
    """Абстрактный класс для обнаружения новых паттернов"""

    @abstractmethod
    async def detect_new_patterns(self, recent_requests: List[UserRequest]) -> List[Dict[str, Any]]:
        """Выявляет новые паттерны в пользовательских запросах"""
