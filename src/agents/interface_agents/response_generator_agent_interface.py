from abc import ABC, abstractmethod
from typing import Dict, Any


class AbstractResponseGeneratorAgent(ABC):
    """Абстрактный класс для генерации ответов"""

    @abstractmethod
    async def generate_response(self, execution_results: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Генерирует финальный ответ пользователю на основе результатов выполнения"""