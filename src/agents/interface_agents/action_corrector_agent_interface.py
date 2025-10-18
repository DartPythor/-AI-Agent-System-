from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import ActionPlan
from typing import List


class AbstractActionCorrectorAgent(ABC):
    """Абстрактный класс для коррекции небезопасных действий"""

    @abstractmethod
    async def correct_unsafe_actions(self, action_plan: ActionPlan, safety_issues: List[str]) -> ActionPlan:
        """Корректирует действия для устранения проблем безопасности"""
