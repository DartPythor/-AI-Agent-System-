from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import IncidentInfo
from typing import List


class AbstractMassIncidentManagerAgent(ABC):
    """Абстрактный класс для управления массовыми инцидентами"""

    @abstractmethod
    async def handle_mass_incident(self, incident: IncidentInfo) -> None:
        """Управляет массовым инцидентом: уведомления, статус страницы"""

    @abstractmethod
    async def broadcast_update(self, message: str, affected_users: List[str]) -> None:
        """Рассылает уведомления пользователям"""
