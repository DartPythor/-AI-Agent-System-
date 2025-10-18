from abc import ABC, abstractmethod
from typing import List, Dict, Any


class AbstractKnowledgeRetrievalAgent(ABC):
    """Абстрактный класс для извлечения знаний из базы"""

    @abstractmethod
    async def retrieve_knowledge(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Извлекает релевантную информацию из базы знаний"""
