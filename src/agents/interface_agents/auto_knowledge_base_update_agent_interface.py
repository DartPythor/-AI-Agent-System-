from abc import ABC, abstractmethod
from typing import List, Dict, Any


class AbstractAutoKnowledgeBaseUpdateAgent(ABC):
    """Абстрактный класс для автоматического обновления базы знаний"""

    @abstractmethod
    async def update_knowledge_base(self, new_patterns: List[Dict[str, Any]]) -> None:
        """Автоматически обновляет базу знаний на основе новых паттернов"""
