from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.agents.interface_agents.agent_response import ToolInvocationResult


class AbstractToolInvocationManagerAgent(ABC):
    """Абстрактный класс для управления вызовом инструментов"""

    @abstractmethod
    async def invoke_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolInvocationResult:
        """Вызывает внешний инструмент с указанными параметрами"""

    @abstractmethod
    async def get_available_tools(self) -> List[str]:
        """Возвращает список доступных инструментов"""
