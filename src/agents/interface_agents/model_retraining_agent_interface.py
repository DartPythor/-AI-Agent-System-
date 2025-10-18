from abc import ABC, abstractmethod
from typing import Dict, Any


class AbstractModelRetrainingAgent(ABC):
    """Абстрактный класс для переобучения моделей"""

    @abstractmethod
    async def retrain_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Организует переобучение ML-моделей на новых данных"""
