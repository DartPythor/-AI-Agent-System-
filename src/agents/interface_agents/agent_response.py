from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime


class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyStatus(str, Enum):
    SAFE = "safe"
    TOXIC = "toxic"
    UNCERTAIN = "uncertain"


class IncidentType(str, Enum):
    SINGLE = "single"
    MASS = "mass"
    SYSTEM_FAILURE = "system_failure"


class Response(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность классификации от 0 до 1")
    classification: str = Field(min_length=1, description="Классификация запроса")
    message: Optional[str] = Field(description="Дополнительное сообщение")

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v


class SafetyCheckResult(BaseModel):
    status: SafetyStatus = Field(description="Статус безопасности")
    score: float = Field(ge=0.0, le=1.0, description="Оценка безопасности от 0 до 1")
    reasons: List[str] = Field(default_factory=list, description="Причины принятия решения")

    @validator('score')
    def validate_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


class UserRequest(BaseModel):
    user_id: str = Field(min_length=1, description="Идентификатор пользователя")
    message: str = Field(min_length=1, description="Текст запроса")
    user_role: str = Field(min_length=1, description="Роль пользователя")
    timestamp: datetime = Field(default_factory=datetime.now, description="Временная метка запроса")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IncidentInfo(BaseModel):
    incident_id: str = Field(min_length=1, description="Идентификатор инцидента")
    incident_type: IncidentType = Field(description="Тип инцидента")
    affected_users: int = Field(ge=0, description="Количество затронутых пользователей")
    severity: PriorityLevel = Field(description="Уровень серьезности")

    @validator('affected_users')
    def validate_affected_users(cls, v):
        if v < 0:
            raise ValueError('Affected users cannot be negative')
        return v


class ActionPlan(BaseModel):
    actions: List[str] = Field(description="Список действий для выполнения")
    tools_required: List[str] = Field(default_factory=list, description="Требуемые инструменты")
    estimated_duration: int = Field(ge=0, description="Оценочная длительность в секундах")
    safety_risk: float = Field(ge=0.0, le=1.0, description="Уровень риска безопасности")

    @validator('estimated_duration')
    def validate_duration(cls, v):
        if v < 0:
            raise ValueError('Estimated duration cannot be negative')
        return v

    @validator('safety_risk')
    def validate_safety_risk(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Safety risk must be between 0 and 1')
        return v


class HealthMetrics(BaseModel):
    automation_rate: float = Field(ge=0.0, le=1.0, description="Уровень автоматизации")
    triage_accuracy: float = Field(ge=0.0, le=1.0, description="Точность триажирования")
    sla_compliance: float = Field(ge=0.0, le=1.0, description="Соблюдение SLA")
    queue_depth: int = Field(ge=0, description="Глубина очереди")
    oldest_task_age: int = Field(ge=0, description="Возраст самой старой задачи в секундах")
    csat_score: float = Field(ge=0.0, le=1.0, description="Оценка удовлетворенности клиентов")

    @validator('automation_rate', 'triage_accuracy', 'sla_compliance', 'csat_score')
    def validate_percentage_fields(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Percentage fields must be between 0 and 1')
        return v

    @validator('queue_depth', 'oldest_task_age')
    def validate_non_negative_fields(cls, v):
        if v < 0:
            raise ValueError('Queue depth and task age cannot be negative')
        return v


class ToolInvocationResult(BaseModel):
    success: bool = Field(description="Успешность выполнения")
    result: Any = Field(description="Результат выполнения")
    error_message: Optional[str] = Field(description="Сообщение об ошибке")
    execution_time: float = Field(0.0, ge=0.0, description="Время выполнения в секундах")

    @validator('execution_time')
    def validate_execution_time(cls, v):
        if v < 0:
            raise ValueError('Execution time cannot be negative')
        return v


class ToolParameter(BaseModel):
    name: str = Field(description="Название параметра")
    type: str = Field(description="Тип параметра")
    required: bool = Field(True, description="Обязательность параметра")
    description: Optional[str] = Field(description="Описание параметра")


class ToolDefinition(BaseModel):
    name: str = Field(description="Название инструмента")
    description: str = Field(description="Описание инструмента")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Параметры инструмента")
    category: str = Field(description="Категория инструмента")


class FeedbackData(BaseModel):
    user_id: str = Field(description="Идентификатор пользователя")
    conversation_id: str = Field(description="Идентификатор диалога")
    rating: int = Field(ge=1, le=5, description="Оценка от 1 до 5")
    comments: Optional[str] = Field(description="Комментарии пользователя")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время оставления отзыва")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemAlert(BaseModel):
    alert_id: str = Field(description="Идентификатор алерта")
    alert_type: str = Field(description="Тип алерта")
    severity: PriorityLevel = Field(description="Уровень серьезности")
    message: str = Field(description="Сообщение алерта")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания алерта")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
