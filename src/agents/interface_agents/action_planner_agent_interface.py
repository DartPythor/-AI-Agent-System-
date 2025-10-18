from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import Response, ActionPlan
from typing import Dict, Any


class AbstractActionPlannerAgent(ABC):
    """Абстрактный класс для планирования действий"""

    @abstractmethod
    async def create_action_plan(self, intent: Response, context: Dict[str, Any]) -> ActionPlan:
        """Создает план действий на основе намерения и контекста"""
