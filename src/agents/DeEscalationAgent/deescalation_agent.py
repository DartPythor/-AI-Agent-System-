import logging
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agents.interface_agents.de_escalation_agent_interface import AbstractDeEscalationAgent

logger = logging.getLogger(__name__)


class DeescalatedResponse(BaseModel):
    """Модель для структурированного ответа деэскалации"""
    deescalated_text: str = Field(description="Ответ для пользователя, чтобы он дал больше информации об проблеме")


class DeEscalationAgent(AbstractDeEscalationAgent):
    """Реализация агента для деэскалации конфликтных ситуаций"""

    def __init__(
            self,
            llm: ChatOpenAI,
            model_name: str = "gpt-4",
            temperature: float = 0.3
    ):
        """
        Args:
            llm: Языковая модель для деэскалации
            model_name: Название модели
            temperature: Температура для генерации (ниже = более консервативно)
        """
        self._model = llm
        self._model_name = model_name
        self._temperature = temperature
        self._struct_model = llm.with_structured_output(DeescalatedResponse)
        self._prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Создает шаблон промпта для деэскалации"""

        system_message = SystemMessage(content="""Вы - эксперт в техподдержки, тебе дают текст, пользователь сильно переживает и грубит, попросили его в вежливо описать проблему, чтобы ему помочь.""")

        return ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content="{toxic_text}")
        ])

    async def deescalate_conversation(self, toxic_text: str) -> str:
        """
        Преобразует токсичный запрос в нейтральный и конструктивный

        Args:
            toxic_text: Исходный токсичный текст

        Returns:
            str: Деэскалированная версия текста
        """
        try:
            if not toxic_text or not toxic_text.strip():
                return "Пожалуйста, опишите вашу проблему, и я постараюсь помочь."

            chain = self._prompt_template | self._struct_model

            result: DeescalatedResponse = await chain.ainvoke({
                "toxic_text": toxic_text
            })

            logger.info(
                f"Deescalation completed."
            )

            return result.deescalated_text

        except Exception as e:
            logger.error(f"Error in deescalation: {e}")
            return self._fallback_deescalation(toxic_text)


async def main():
    import logging
    from langchain_openai import ChatOpenAI

    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    # Инициализация модели (в реальном проекте настройки берутся из конфига)
    llm = ChatOpenAI(
        model="mistralai/mistral-small-3.2-24b-instruct",
        temperature=0.3,
        api_key="sk-or-vv-264b7ac948300c5bd342c7fe83339dd3b38a269668c36e4fbad3fca8ee859345",
        base_url="https://api.vsegpt.ru/v1",
        max_tokens=1000,
    )

    # Создание агента деэскалации
    deescalation_agent = DeEscalationAgent(llm=llm)

    # Тестовые примеры токсичных сообщений
    test_messages = [
        "Вы все совершенно некомпетентны! Почему система постоянно падает?",
        "Мне надоело это дерьмо! Когда уже почините эту хрень?!",
        "Я требую немедленно решить проблему! Иначе я буду жаловаться!",
        "Ваша поддержка - просто шутка! Никто не может нормально помочь!",
        "Почему я должен ждать? Вы что, не понимаете, что у меня срочная работа?",
    ]

    for toxic_message in test_messages:
        print(f"\n{'=' * 50}")
        print(f"Исходное сообщение: {toxic_message}")

        deescalated = await deescalation_agent.deescalate_conversation(toxic_message)
        print(f"Деэскалированное: {deescalated}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
