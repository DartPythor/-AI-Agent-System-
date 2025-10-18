import logging
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agents.interface_agents.response_generator_agent_interface import (
    AbstractResponseGeneratorAgent, GeneratedResponse, ResponseContext
)

logger = logging.getLogger(__name__)


class StructuredResponse(BaseModel):
    """Структурированный ответ для LLM"""
    response_text: str = Field(..., description="Основной текст ответа пользователю")
    response_type: str = Field(..., description="information, instruction, escalation, clarification, confirmation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность в ответе")
    suggested_actions: List[str] = Field(default_factory=list, description="Конкретные действия для пользователя")
    next_steps: List[str] = Field(default_factory=list, description="Что будет делать система")
    requires_human_review: bool = Field(False, description="Требуется ли проверка человеком")


class ResponseGeneratorAgent(AbstractResponseGeneratorAgent):
    """Реализация агента для генерации ответов пользователю"""

    def __init__(
            self,
            llm: ChatOpenAI,
            model_name: str = "gpt-4",
            temperature: float = 0.3
    ):
        """
        Args:
            llm: Языковая модель для генерации ответов
            model_name: Название модели
            temperature: Температура для генерации
        """
        self._model = llm
        self._model_name = model_name
        self._temperature = temperature
        self._struct_model = llm.with_structured_output(StructuredResponse)
        self.prompt = ChatPromptTemplate([])
        self._set_prompt()

    def _set_prompt(self):
        """Устанавливает системный промпт для генерации ответов"""
        self.prompt.append(SystemMessage(
            content="""Вы - профессиональный ассистент технической поддержки. Ваша задача - генерировать полезные, точные и структурированные ответы для пользователей на основе предоставленной информации.

КОНТЕКСТ ОТВЕТА:
- User Intent: Намерение пользователя (техническая поддержка, биллинг, управление аккаунтом и т.д.)
- Retrieved Knowledge: Информация, извлеченная из базы знаний
- User Context: Роль пользователя и другая контекстная информация

ПРИНЦИПЫ ГЕНЕРАЦИИ ОТВЕТОВ:
1. Будьте полезным и релевантным - отвечайте именно на вопрос пользователя
2. Будьте точным - используйте только предоставленную информацию из базы знаний
3. Будьте структурированным - организуйте ответ логически
4. Будьте вежливым - используйте профессиональный и уважительный тон
5. Будьте прозрачным - если информации недостаточно, честно сообщите об этом

ТИПЫ ОТВЕТОВ:
- information: Предоставление информации из базы знаний
- instruction: Пошаговые инструкции для решения проблемы
- escalation: Эскалация к человеку-оператору
- clarification: Запрос дополнительной информации
- confirmation: Подтверждение выполненного действия

СТРУКТУРА ОТВЕТА:
1. Приветствие и подтверждение понимания запроса
2. Основная информация/решение
3. Конкретные шаги (если применимо)
4. Дополнительные рекомендации
5. Предложение дальнейшей помощи

ПРАВИЛА ДЛЯ РАЗНЫХ СИТУАЦИЙ:
- Если уверенность низкая (<0.7) - запросите уточнение или предложите эскалацию
- Если проблема критическая - немедленно предлагайте эскалацию
- Если информация найдена - предоставьте четкие инструкции
- Если информации нет - честно сообщите и предложите альтернативы

ФОРМАТ СТРУКТУРИРОВАННОГО ОТВЕТА:
- response_text: Полный текст ответа пользователю
- response_type: Тип ответа (information/instruction/escalation/clarification/confirmation)
- confidence: Уверенность от 0.0 до 1.0
- suggested_actions: Конкретные действия для пользователя
- next_steps: Что система сделает дальше
- requires_human_review: true/false (если уверенность < 0.7 или проблема критическая)"""
        ))

    def _set_context(self, response_context: ResponseContext):
        """Добавляет контекст в промпт"""
        # Форматируем извлеченные знания
        knowledge_text = ""
        if response_context.retrieved_knowledge.get('documents'):
            for i, doc in enumerate(response_context.retrieved_knowledge['documents'][:3], 1):
                knowledge_text += f"Документ {i}: {doc.get('content', '')[:200]}...\n"

        # Форматируем историю диалога
        history_text = ""
        if response_context.conversation_history:
            for msg in response_context.conversation_history[-3:]:  # Последние 3 сообщения
                role = "Пользователь" if msg.get('role') == 'user' else "Ассистент"
                history_text += f"{role}: {msg.get('content', '')}\n"

        context_message = f"""
КОНТЕКСТ ДЛЯ ГЕНЕРАЦИИ ОТВЕТА:

Намерение пользователя: {response_context.user_intent}
Оригинальный запрос: {response_context.user_query}
Роль пользователя: {response_context.user_context.get('user_role', 'Неизвестно')}

ИЗВЛЕЧЕННЫЕ ЗНАНИЯ:
{knowledge_text if knowledge_text else 'Информация не найдена в базе знаний'}

ИСТОРИЯ ДИАЛОГА:
{history_text if history_text else 'История отсутствует'}

ВОЗМОЖНОСТИ СИСТЕМЫ:
{', '.join(response_context.system_capabilities) if response_context.system_capabilities else 'Базовые возможности'}

Сгенерируйте ответ на основе этой информации:"""

        self.prompt.append(HumanMessage(content=context_message))

    def delete_context(self):
        """Очищает контекст из промпта"""
        self.prompt = ChatPromptTemplate([])
        self._set_prompt()

    async def generate_response(self, execution_results: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Генерирует финальный ответ пользователю на основе результатов выполнения

        Args:
            execution_results: Результаты выполнения действий
            context: Контекст выполнения

        Returns:
            str: Текст ответа пользователю
        """
        try:
            # Создаем контекст ответа из execution_results и context
            response_context = self._create_response_context(execution_results, context)

            # Генерируем структурированный ответ
            structured_response = await self.generate_structured_response(response_context)

            logger.info(
                f"Response generated. Type: {structured_response.response_type}, "
                f"Confidence: {structured_response.confidence:.2f}, "
                f"Human review: {structured_response.requires_human_review}"
            )

            return structured_response.response_text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(execution_results, context, str(e))

    def _create_response_context(self, execution_results: Dict[str, Any], context: Dict[str, Any]) -> ResponseContext:
        """Создает контекст ответа из execution_results и context"""
        return ResponseContext(
            user_intent=execution_results.get('intent', {}).get('classification', 'unknown'),
            user_query=context.get('user_query', ''),
            retrieved_knowledge=execution_results.get('knowledge', {}),
            user_context=context.get('user_context', {}),
            conversation_history=context.get('conversation_history', []),
            system_capabilities=context.get('system_capabilities', [])
        )

    async def generate_structured_response(self, response_context: ResponseContext) -> GeneratedResponse:
        """
        Генерирует структурированный ответ с метаданными

        Args:
            response_context: Контекст для генерации ответа

        Returns:
            GeneratedResponse: Структурированный ответ
        """
        try:
            # Добавляем контекст в промпт
            self._set_context(response_context)

            # Генерируем структурированный ответ
            result: StructuredResponse = self._struct_model.invoke(str(self.prompt))

            # Очищаем контекст
            self.delete_context()

            # Преобразуем в GeneratedResponse
            return GeneratedResponse(
                response_text=result.response_text,
                response_type=result.response_type,
                confidence=result.confidence,
                suggested_actions=result.suggested_actions,
                next_steps=result.next_steps,
                requires_human_review=result.requires_human_review,
                metadata={
                    "generation_timestamp": "current_timestamp",  # В реальности datetime.now().isoformat()
                    "model_used": self._model_name,
                    "user_intent": response_context.user_intent
                }
            )

        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            self.delete_context()
            return self._generate_fallback_structured_response(response_context, str(e))

    def _generate_fallback_structured_response(self, response_context: ResponseContext,
                                               error: str) -> GeneratedResponse:
        """Генерирует резервный структурированный ответ при ошибке"""
        # Анализируем контекст для создания базового ответа
        user_intent = response_context.user_intent
        has_knowledge = bool(response_context.retrieved_knowledge.get('documents'))

        if has_knowledge:
            response_text = self._create_basic_response_from_knowledge(response_context)
            confidence = 0.6
        else:
            response_text = self._create_escalation_response(response_context)
            confidence = 0.3

        return GeneratedResponse(
            response_text=response_text,
            response_type="escalation" if not has_knowledge else "information",
            confidence=confidence,
            suggested_actions=["Обратиться к специалисту поддержки"] if not has_knowledge else [
                "Следовать инструкциям"],
            next_steps=["Эскалация к оператору"] if not has_knowledge else ["Завершение обработки запроса"],
            requires_human_review=not has_knowledge,
            metadata={"error": error, "fallback": True}
        )

    def _create_basic_response_from_knowledge(self, response_context: ResponseContext) -> str:
        """Создает базовый ответ из извлеченных знаний"""
        docs = response_context.retrieved_knowledge.get('documents', [])
        if not docs:
            return self._create_escalation_response(response_context)

        # Берем первый документ как основной источник
        main_doc = docs[0]
        content = main_doc.get('content', '')[:300]  # Ограничиваем длину

        return f"""Здравствуйте! На основе информации из базы знаний могу сообщить:

{content}

Для получения более подробной информации или помощи в решении вашей проблемы, пожалуйста, обратитесь к специалисту поддержки."""

    def _create_escalation_response(self, response_context: ResponseContext) -> str:
        """Создает ответ с эскалацией"""
        return f"""Здравствуйте! К сожалению, я не смог найти достаточно информации в базе знаний для ответа на ваш вопрос о "{response_context.user_query}".

Я передам ваш запрос специалисту поддержки, который свяжется с вами в ближайшее время.

Для ускорения решения проблемы вы можете:
- Уточнить детали вашего запроса
- Обратиться напрямую по телефону поддержки: +7 (XXX) XXX-XX-XX
- Написать на email: support@company.com"""

    def _generate_fallback_response(self, execution_results: Dict[str, Any], context: Dict[str, Any],
                                    error: str) -> str:
        """Генерирует резервный ответ при ошибке"""
        user_query = context.get('user_query', 'ваш запрос')

        return f"""Здравствуйте! В настоящее время я испытываю технические трудности с обработкой вашего запроса.

Ваш вопрос: "{user_query}"

Пожалуйста, попробуйте:
1. Повторить запрос через несколько минут
2. Обратиться в поддержку по телефону: +7 (XXX) XXX-XX-XX
3. Написать на email: support@company.com

Приносим извинения за временные неудобства."""

    async def generate_follow_up_questions(self, response_context: ResponseContext) -> List[str]:
        """
        Генерирует уточняющие вопросы для продолжения диалога

        Args:
            response_context: Контекст для генерации вопросов

        Returns:
            List[str]: Список уточняющих вопросов
        """
        try:
            follow_up_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Сгенерируйте 2-3 уточняющих вопроса на основе контекста диалога.
Вопросы должны помочь:
1. Уточнить детали проблемы
2. Получить дополнительную информацию
3. Направить пользователя к решению

Будьте конкретными и полезными."""),
                HumanMessage(content=f"""
Запрос пользователя: {response_context.user_query}
Намерение: {response_context.user_intent}
Найденная информация: {response_context.retrieved_knowledge.get('answer', 'Информация не найдена')}

Сгенерируйте уточняющие вопросы:""")
            ])

            follow_up_chain = follow_up_prompt | self._model
            result = await follow_up_chain.ainvoke({})

            # Парсим вопросы из ответа
            questions = self._parse_questions_from_text(result.content)
            return questions[:3]  # Ограничиваем 3 вопросами

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return [
                "Можете уточнить детали вашей проблемы?",
                "Какие шаги вы уже предприняли для решения?",
                "Есть ли дополнительная информация, которая может помочь?"
            ]

    def _parse_questions_from_text(self, text: str) -> List[str]:
        """Парсит вопросы из текста"""
        import re
        # Ищем вопросы (строки, заканчивающиеся на ? или начинающиеся с цифр)
        questions = re.findall(r'(?:\d+\.\s*)?([^.!?]*\?)', text)
        return [q.strip() for q in questions if q.strip()]

    async def generate_response_for_intent(self, intent: str, knowledge_result: Dict[str, Any],
                                           user_context: Dict[str, Any]) -> GeneratedResponse:
        """
        Генерирует ответ для конкретного интента

        Args:
            intent: Намерение пользователя
            knowledge_result: Результат извлечения знаний
            user_context: Контекст пользователя

        Returns:
            GeneratedResponse: Сгенерированный ответ
        """
        response_context = ResponseContext(
            user_intent=intent,
            user_query=user_context.get('original_query', ''),
            retrieved_knowledge=knowledge_result,
            user_context=user_context,
            conversation_history=[],
            system_capabilities=["knowledge_retrieval", "basic_instructions"]
        )

        return await self.generate_structured_response(response_context)

    async def evaluate_response_quality(self, response: GeneratedResponse,
                                        user_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Оценивает качество сгенерированного ответа

        Args:
            response: Сгенерированный ответ
            user_feedback: Обратная связь от пользователя

        Returns:
            Dict: Оценка качества ответа
        """
        quality_metrics = {
            "confidence": response.confidence,
            "response_length": len(response.response_text),
            "has_suggested_actions": len(response.suggested_actions) > 0,
            "requires_human_review": response.requires_human_review,
            "response_type_appropriate": self._check_response_type_appropriateness(response)
        }

        if user_feedback:
            quality_metrics["user_rating"] = user_feedback.get('rating')
            quality_metrics["user_comments"] = user_feedback.get('comments')

        # Вычисляем общую оценку
        quality_score = self._calculate_quality_score(quality_metrics)
        quality_metrics["overall_quality_score"] = quality_score

        return quality_metrics

    def _check_response_type_appropriateness(self, response: GeneratedResponse) -> bool:
        """Проверяет уместность типа ответа"""
        inappropriate_escalation = (
                response.response_type == "escalation" and
                response.confidence > 0.8 and
                len(response.response_text) < 50
        )

        return not inappropriate_escalation

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Вычисляет общую оценку качества"""
        score = 0.0
        weights = {
            "confidence": 0.4,
            "has_suggested_actions": 0.2,
            "response_type_appropriate": 0.3,
            "user_rating": 0.1 if "user_rating" in metrics else 0.0
        }

        if metrics["confidence"]:
            score += metrics["confidence"] * weights["confidence"]

        if metrics["has_suggested_actions"]:
            score += weights["has_suggested_actions"]

        if metrics["response_type_appropriate"]:
            score += weights["response_type_appropriate"]

        if "user_rating" in metrics and metrics["user_rating"]:
            score += (metrics["user_rating"] / 5) * weights["user_rating"]

        return min(score, 1.0)

    def update_response_templates(self, new_templates: Dict[str, str]):
        """
        Обновляет шаблоны ответов

        Args:
            new_templates: Новые шаблоны ответов
        """
        # В реальной системе здесь будет обновление промпта с новыми шаблонами
        logger.info(f"Updated response templates with {len(new_templates)} new templates")