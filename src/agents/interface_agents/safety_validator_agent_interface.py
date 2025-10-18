from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import ActionPlan, SafetyCheckResult


class AbstractSafetyValidatorAgent(ABC):
    """Абстрактный класс для валидации безопасности действий"""

    @abstractmethod
    async def validate_action_safety(self, action_plan: ActionPlan) -> SafetyCheckResult:
        """Проверяет безопасность запланированных действий"""
