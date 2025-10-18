import logging
from typing import List, Dict, Any
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agents.interface_agents.agent_response import Response
from src.agents.interface_agents.intent_classifier_agent_interface import AbstractIntentClassifierAgent

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Типы намерений пользователя"""
    TECHNICAL_SUPPORT = "technical_support"
    BILLING = "billing"
    ACCOUNT_MANAGEMENT = "account_management"
    ACCESS_REQUEST = "access_request"
    SYSTEM_ISSUE = "system_issue"
    FEATURE_REQUEST = "feature_request"
    COMPLAINT = "complaint"
    STATUS_CHECK = "status_check"
    PASSWORD_RESET = "password_reset"
    GENERAL_INQUIRY = "general_inquiry"
    EMERGENCY = "emergency"
    DOCUMENTATION = "documentation"


class IntentClassification(BaseModel):
    """Структурированный ответ классификатора намерений"""
    primary_intent: IntentType = Field(description="Основное намерение пользователя")
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в классификации")
    alternative_intents: List[IntentType] = Field(default_factory=list, description="Альтернативные намерения")
    reasoning: str = Field(description="Обоснование классификации")
    requires_immediate_attention: bool = Field(description="Требует немедленного внимания")


class IntentClassifierAgent(AbstractIntentClassifierAgent):
    """Реализация агента для классификации намерений пользователя"""

    def __init__(
            self,
            llm: ChatOpenAI,
            model_name: str = "gpt-4",
            temperature: float = 0.1,
            confidence_threshold: float = 0.7
    ):
        """
        Args:
            llm: Языковая модель для классификации
            model_name: Название модели
            temperature: Температура для генерации (ниже = более консервативно)
            confidence_threshold: Порог уверенности для принятия решения
        """
        self._model = llm
        self._model_name = model_name
        self._temperature = temperature
        self._confidence_threshold = confidence_threshold
        self._struct_model = llm.with_structured_output(IntentClassification)
        self.prompt = ChatPromptTemplate([])
        self._set_prompt()

    def _set_prompt(self):
        """Устанавливает системный промпт для классификации"""
        self.prompt.append(SystemMessage(
            content="""Вы - эксперт по классификации намерений в системе технической поддержки.
Ваша задача - точно определить тип запроса пользователя на основе предоставленного сообщения.

ДОСТУПНЫЕ КАТЕГОРИИ НАМЕРЕНИЙ:

1. technical_support - Проблемы с программным обеспечением, оборудованием, ошибки в работе
   Примеры: "не запускается программа", "ошибка при подключении", "не работает принтер"

2. billing - Вопросы, связанные с оплатой, счетами, тарифами
   Примеры: "не пришел счет", "опрос по оплате подписки", "изменить тарифный план"

3. account_management - Управление учетной записью, настройки профиля
   Примеры: "сменить email", "обновить личные данные", "настройки уведомлений"

4. access_request - Запросы на доступ к системам, приложениям, данным
   Примеры: "нужен доступ к SAP", "предоставить права на папку", "доступ к базе данных"

5. system_issue - Проблемы с системной инфраструктурой, сетью, серверами
   Примеры: "медленная сеть", "недоступен сервер", "проблемы с VPN"

6. feature_request - Запросы на новые функции или улучшения
   Примеры: "хочу новую функцию в системе", "можно добавить отчетность", "предложение по улучшению"

7. complaint - Жалобы на сервис, качество обслуживания
   Примеры: "недоволен работой поддержки", "жалоба на сотрудника", "плохое качество связи"

8. status_check - Проверка статуса заявки, задачи, процесса
   Примеры: "какой статус моей заявки", "когда будет готово", "отследить прогресс"

9. password_reset - Сброс или восстановление пароля
   Примеры: "не могу войти в систему", "сбросить пароль", "забыл пароль"

10. general_inquiry - Общие вопросы, не относящиеся к другим категориям
    Примеры: "часы работы отдела", "контактная информация", "общие сведения о услуге"

11. emergency - Критические сбои, требующие немедленного вмешательства
    Примеры: "полностью недоступна система", "авария на производстве", "критический сбой"

12. documentation - Запросы документации, инструкций, руководств
    Примеры: "где найти инструкцию", "нужна документация по API", "руководство пользователя"

ПРАВИЛА КЛАССИФИКАЦИИ:
- Выбирайте наиболее специфичную категорию
- Если запрос подходит под несколько категорий, выберите основную и укажите альтернативные
- Отмечайте requires_immediate_attention=true для emergency и критических system_issue
- Будьте внимательны к контексту и срочности

ФОРМАТ ОТВЕТА:
- primary_intent: основная категория
- confidence: уверенность от 0.0 до 1.0
- alternative_intents: альтернативные категории (если есть)
- reasoning: краткое обоснование выбора
- requires_immediate_attention: true/false"""
        ))

    def _set_text(self, text: str):
        """Добавляет пользовательский запрос в промпт"""
        self.prompt.append(
            HumanMessage(content=f"Классифицируйте следующий запрос пользователя: {text}"),
        )

    def delete_text(self):
        """Очищает промпт от пользовательских сообщений"""
        self.prompt = ChatPromptTemplate([])
        self._set_prompt()

    async def get_classification(self, data: str) -> Response:
        """
        Возвращает классификацию запроса и уверенность в возможности дать ответ

        Args:
            data: Текст запроса пользователя

        Returns:
            Response: Классификация и уверенность
        """
        try:
            if not data or not data.strip():
                return Response(
                    confidence=0.0,
                    classification=IntentType.GENERAL_INQUIRY.value,
                    message="Пустой запрос"
                )

            self._set_text(data)

            result: IntentClassification = self._struct_model.invoke(str(self.prompt))

            self.delete_text()

            logger.info(
                f"Intent classification completed. "
                f"Intent: {result.primary_intent.value}, "
                f"Confidence: {result.confidence:.2f}, "
                f"Immediate attention: {result.requires_immediate_attention}"
            )

            message = f"Классифицировано как {result.primary_intent.value}. {result.reasoning}"
            if result.alternative_intents:
                alt_intents = ", ".join([intent.value for intent in result.alternative_intents])
                message += f" Альтернативные варианты: {alt_intents}."
            if result.requires_immediate_attention:
                message += " ТРЕБУЕТ НЕМЕДЛЕННОГО ВНИМАНИЯ."

            return Response(
                confidence=result.confidence,
                classification=result.primary_intent.value,
                message=message
            )

        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            self.delete_text()
            return self._fallback_classification(data, str(e))

    def _fallback_classification(self, data: str, error: str) -> Response:
        """
        Базовая классификация на случай ошибки LLM

        Args:
            data: Текст запроса
            error: Сообщение об ошибке

        Returns:
            Response: Базовая классификация
        """
        data_lower = data.lower()

        keyword_mapping = {
            IntentType.PASSWORD_RESET: ['парол', 'войти', 'логин', 'авторизац'],
            IntentType.TECHNICAL_SUPPORT: ['не работает', 'ошибка', 'сломал', 'глюк', 'принтер', 'ремонт'],
            IntentType.BILLING: ['оплат', 'счет', 'тариф', 'деньги'],
            IntentType.SYSTEM_ISSUE: ['сеть', 'сервер', 'интернет', 'vpn'],
            IntentType.ACCESS_REQUEST: ['доступ', 'права', 'разрешен'],
            IntentType.STATUS_CHECK: ['статус', 'когда', 'готов', 'проверить'],
            IntentType.EMERGENCY: ['срочно', 'авария', 'критич', 'недоступен']
        }

        for intent_type, keywords in keyword_mapping.items():
            if any(keyword in data_lower for keyword in keywords):
                return Response(
                    confidence=0.5,  # Низкая уверенность для fallback
                    classification=intent_type.value,
                    message=f"Базовая классификация. Ошибка LLM: {error}"
                )

        return Response(
            confidence=0.3,
            classification=IntentType.GENERAL_INQUIRY.value,
            message=f"Не удалось точно классифицировать. Ошибка: {error}"
        )

    async def get_detailed_classification(self, data: str) -> Dict[str, Any]:
        """
        Расширенная классификация с детализацией

        Args:
            data: Текст запроса пользователя

        Returns:
            Dict: Детализированные результаты классификации
        """
        try:
            detailed_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Проанализируйте запрос пользователя и предоставьте детализированную классификацию.
Укажите:
1. Основную категорию
2. Подкатегорию (если применимо)
3. Уровень срочности (низкий, средний, высокий, критический)
4. Ключевые слова, повлиявшие на классификацию
5. Рекомендуемый путь обработки"""),
                HumanMessage(content=f"Запрос для анализа: {data}")
            ])

            detailed_chain = detailed_prompt | self._model
            detailed_result = await detailed_chain.ainvoke({})

            main_classification = await self.get_classification(data)

            return {
                "main_classification": {
                    "intent": main_classification.classification,
                    "confidence": main_classification.confidence,
                    "message": main_classification.message
                },
                "detailed_analysis": detailed_result.content,
                "timestamp": "current_timestamp",
                "requires_human_review": main_classification.confidence < self._confidence_threshold
            }

        except Exception as e:
            logger.error(f"Error in detailed classification: {e}")
            main_classification = await self.get_classification(data)
            return {
                "main_classification": {
                    "intent": main_classification.classification,
                    "confidence": main_classification.confidence,
                    "message": main_classification.message
                },
                "detailed_analysis": "Детальный анализ недоступен",
                "error": str(e),
                "requires_human_review": True
            }

    def update_intent_categories(self, new_categories: Dict[str, List[str]]) -> None:
        """
        Обновляет категории намерений новыми примерами

        Args:
            new_categories: Словарь с новыми категориями и примерами
                {intent_type: [пример1, пример2]}
        """
        try:
            examples_section = "\n\nДОПОЛНИТЕЛЬНЫЕ ПРИМЕРЫ КЛАССИФИКАЦИИ:\n"
            for intent_type, examples in new_categories.items():
                examples_section += f"\n{intent_type}:\n"
                for example in examples:
                    examples_section += f"  - {example}\n"
            current_content = self.prompt.messages[0].content
            if "ДОПОЛНИТЕЛЬНЫЕ ПРИМЕРЫ КЛАССИФИКАЦИИ:" in current_content:
                current_content = current_content.split("ДОПОЛНИТЕЛЬНЫЕ ПРИМЕРЫ КЛАССИФИКАЦИИ:")[0]

            updated_content = current_content + examples_section
            self.prompt.messages[0] = SystemMessage(content=updated_content)

            logger.info(f"Updated intent categories with {sum(len(v) for v in new_categories.values())} new examples")

        except Exception as e:
            logger.error(f"Error updating intent categories: {e}")

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Устанавливает порог уверенности для классификации

        Args:
            threshold: Новый порог уверенности (0.0 - 1.0)
        """
        if 0 <= threshold <= 1:
            self._confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0 and 1")


# Настройка логирования
logging.basicConfig(level=logging.INFO)
import asyncio


async def main():
    # Инициализация модели
    llm = ChatOpenAI(
        model="deepseek/deepseek-v3.2-exp-alt",
        temperature=0.1,
        api_key="sk-or-vv-264b7ac948300c5bd342c7fe83339dd3b38a269668c36e4fbad3fca8ee859345",
        base_url="https://api.vsegpt.ru/v1",
    )

    # Создание агента классификации
    intent_classifier = IntentClassifierAgent(llm=llm, confidence_threshold=0.7)

    # Тестовые запросы
    test_queries = [
        "Сломался принтер, вызовите ремонтика.",
        "Ужас, взломали систему!",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Запрос: {query}")

        # Классификация
        result = await intent_classifier.get_classification(query)
        print(f"Намерение: {result.classification}")
        print(f"Уверенность: {result.confidence:.2f}")
        print(f"Сообщение: {result.message}")

        # Детализированный анализ (опционально)
        if result.confidence < 0.7:
            detailed = await intent_classifier.get_detailed_classification(query)
            print(f"Требует проверки человеком: {detailed['requires_human_review']}")


if __name__ == "__main__":
    asyncio.run(main())