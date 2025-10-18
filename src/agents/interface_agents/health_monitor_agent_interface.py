from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import HealthMetrics
from typing import Dict, Any


class AbstractHealthMonitorAgent(ABC):
    """Абстрактный класс для мониторинга здоровья системы"""

    @abstractmethod
    async def get_health_metrics(self) -> HealthMetrics:
        """Возвращает ключевые метрики здоровья системы"""

    @abstractmethod
    async def check_sla_compliance(self) -> Dict[str, Any]:
        """Проверяет соблюдение SLA"""
