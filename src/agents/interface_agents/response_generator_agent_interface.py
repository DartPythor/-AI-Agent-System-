from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class GeneratedResponse(BaseModel):
    """Модель для сгенерированного ответа"""
    response_text: str = Field(..., description="Текст ответа пользователю")
    response_type: str = Field(..., description="Тип ответа (information, instruction, escalation, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность в правильности ответа")
    suggested_actions: List[str] = Field(default_factory=list, description="Предлагаемые действия")
    next_steps: List[str] = Field(default_factory=list, description="Следующие шаги")
    requires_human_review: bool = Field(False, description="Требует проверки человеком")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")


class ResponseContext(BaseModel):
    """Контекст для генерации ответа"""
    user_intent: str = Field(..., description="Намерение пользователя")
    user_query: str = Field(..., description="Оригинальный запрос пользователя")
    retrieved_knowledge: Dict[str, Any] = Field(..., description="Извлеченные знания")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="Контекст пользователя")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="История диалога")
    system_capabilities: List[str] = Field(default_factory=list, description="Возможности системы")


class AbstractResponseGeneratorAgent(ABC):
    """Абстрактный класс для генерации ответов"""

    @abstractmethod
    async def generate_response(self, execution_results: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Генерирует финальный ответ пользователю на основе результатов выполнения"""

    @abstractmethod
    async def generate_structured_response(self, response_context: ResponseContext) -> GeneratedResponse:
        """Генерирует структурированный ответ с метаданными"""

    @abstractmethod
    async def generate_follow_up_questions(self, response_context: ResponseContext) -> List[str]:
        """Генерирует уточняющие вопросы для продолжения диалога"""
