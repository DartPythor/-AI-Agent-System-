from abc import ABC, abstractmethod
from src.agents.interface_agents.agent_response import HealthMetrics
from typing import List, Dict, Any


class AbstractAnomalyDetectorAgent(ABC):
    """Абстрактный класс для обнаружения аномалий"""

    @abstractmethod
    async def detect_anomalies(self, system_metrics: HealthMetrics) -> List[Dict[str, Any]]:
        """Выявляет аномалии в работе системы"""
