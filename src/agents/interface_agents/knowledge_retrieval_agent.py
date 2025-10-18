from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class RetrievedDocument(BaseModel):
    """Модель для извлеченного документа из базы знаний"""
    content: str = Field(..., description="Содержание документа")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Метаданные документа")
    similarity_score: float = Field(..., description="Оценка схожести с запросом")
    source: str = Field(..., description="Источник документа")


class KnowledgeRetrievalResult(BaseModel):
    """Результат извлечения знаний из базы"""
    documents: List[RetrievedDocument] = Field(..., description="Извлеченные документы")
    answer: Optional[str] = Field(None, description="Сгенерированный ответ на основе документов")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность в релевантности ответа")
    suggested_actions: List[str] = Field(default_factory=list, description="Предлагаемые действия")
    additional_questions: List[str] = Field(default_factory=list, description="Дополнительные вопросы для уточнения")


class AbstractKnowledgeRetrievalAgent(ABC):
    """Абстрактный класс для извлечения знаний из базы"""

    @abstractmethod
    async def retrieve_knowledge(self, query: str, context: Dict[str, Any]) -> KnowledgeRetrievalResult:
        """Извлекает релевантную информацию из базы знаний"""

    @abstractmethod
    async def search_with_filters(self, query: str, filters: Dict[str, Any], k: int = 5) -> KnowledgeRetrievalResult:
        """Поиск с дополнительными фильтрами"""

    @abstractmethod
    async def get_related_topics(self, query: str) -> List[str]:
        """Возвращает связанные темы для запроса"""
