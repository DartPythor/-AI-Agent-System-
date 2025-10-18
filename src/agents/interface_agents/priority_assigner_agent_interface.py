from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import UserRequest, Response, PriorityLevel


class AbstractPriorityAssignerAgent(ABC):
    """Абстрактный класс для назначения приоритетов"""

    @abstractmethod
    async def assign_priority(self, request: UserRequest, intent: Response) -> PriorityLevel:
        """Определяет приоритет на основе важности вопроса и роли отправителя"""
