from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import UserRequest, IncidentInfo
from typing import List


class AbstractMassSimilarityDetectorAgent(ABC):
    """Абстрактный класс для кластеризации массовых обращений"""

    @abstractmethod
    async def find_similar_incidents(self, current_request: UserRequest) -> List[IncidentInfo]:
        """Находит кластеры похожих обращений для выявления массовых инцидентов"""
