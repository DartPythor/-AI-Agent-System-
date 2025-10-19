import asyncio
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI

from src.agents.DeEscalationAgent.deescalation_agent import DeEscalationAgent
from src.agents.IntentClassIfierAgent.intent_classifier_agent import IntentClassifierAgent
from src.agents.ToxicDetectorAgent.toxic_detector_agent import ToxicityDetectorAgent
from src.agents.KnowledgeBaseManager.knowledge_retrieval_agent import KnowledgeRetrievalAgent
from src.agents.ResponseGeneratorAgent.response_generator_agent import ResponseGeneratorAgent
from src.agents.KnowledgeBaseManager.knowledge_base_manager import KnowledgeBaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechSupportSystem:
    """Прототип системы техподдержки"""

    def __init__(self):
        self.llm = None
        self.toxicity_detector = None
        self.deescalation_agent = None
        self.intent_classifier = None
        self.knowledge_base_manager = None
        self.knowledge_retrieval_agent = None
        self.response_generator = None

    async def initialize(self):
        """Инициализация всех компонентов системы"""
        try:
            self.llm = ChatOpenAI(
                model="deepseek/deepseek-v3.2-exp-alt",
                temperature=0.3,
                api_key="sk-or-vv-264b7ac948300c5bd342c7fe83339dd3b38a269668c36e4fbad3fca8ee859345",
                base_url="https://api.vsegpt.ru/v1",
                max_tokens=1000,
            )

            # Инициализация агентов
            self.toxicity_detector = ToxicityDetectorAgent(
                toxicity_threshold=0.7,
                high_toxicity_threshold=0.9
            )

            self.deescalation_agent = DeEscalationAgent(llm=self.llm)
            self.intent_classifier = IntentClassifierAgent(llm=self.llm)

            # Инициализация базы знаний
            self.knowledge_base_manager = KnowledgeBaseManager(
                data_path="data/data.md",
                vector_store_path="vector_store"
            )

            await self.knowledge_base_manager.load_or_create_knowledge_base()

            self.knowledge_retrieval_agent = KnowledgeRetrievalAgent(
                knowledge_base_manager=self.knowledge_base_manager,
                similarity_threshold=0.3
            )

            self.response_generator = ResponseGeneratorAgent(llm=self.llm)

            logger.info("✅ Система техподдержки успешно инициализирована")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка инициализации системы: {e}")
            return False

    async def process_user_request(self, user_message: str, user_id: str = "test_user") -> Dict[str, Any]:
        """
        Обрабатывает запрос пользователя через полный конвейер

        Args:
            user_message: Сообщение пользователя
            user_id: Идентификатор пользователя

        Returns:
            Dict: Результаты обработки на каждом этапе
        """
        results = {
            "user_message": user_message,
            "user_id": user_id,
            "processing_steps": {}
        }

        try:
            # Шаг 1: Проверка токсичности
            logger.info("🔍 Проверка токсичности...")
            toxicity_result = await self.toxicity_detector.detect_toxicity(user_message)
            results["processing_steps"]["toxicity_check"] = {
                "status": toxicity_result.status.value,
                "score": toxicity_result.score,
                "reasons": toxicity_result.reasons
            }

            # Шаг 2: Деэскалация при необходимости
            if toxicity_result.status.value == "toxic":
                logger.info("🔄 Деэскалация токсичного сообщения...")
                deescalated_response = await self.deescalation_agent.deescalate_conversation(user_message)
                results["processing_steps"]["deescalation"] = {
                    "applied": True,
                    "deescalated_message": deescalated_response
                }
                # Возвращаем деэскалированный ответ сразу
                results["final_response"] = deescalated_response
                results["processing_complete"] = True
                return results

            results["processing_steps"]["deescalation"] = {"applied": False}

            # Шаг 3: Классификация намерения
            logger.info("🎯 Классификация намерения...")
            intent_result = await self.intent_classifier.get_classification(user_message)
            results["processing_steps"]["intent_classification"] = {
                "intent": intent_result.classification,
                "confidence": intent_result.confidence,
                "message": intent_result.message
            }

            # Шаг 4: Поиск в базе знаний
            logger.info("📚 Поиск в базе знаний...")
            knowledge_context = {
                'k': 5,
                'filters': {'intent': intent_result.classification}
            }

            knowledge_result = await self.knowledge_retrieval_agent.retrieve_knowledge(
                query=user_message,
                context=knowledge_context
            )

            results["processing_steps"]["knowledge_retrieval"] = {
                "documents_found": len(knowledge_result.documents),
                "confidence": knowledge_result.confidence,
                "suggested_actions": knowledge_result.suggested_actions
            }

            # Шаг 5: Генерация ответа
            logger.info("🤖 Генерация ответа...")

            # Подготовка контекста для генератора ответов
            execution_results = {
                'intent': {
                    'classification': intent_result.classification,
                    'confidence': intent_result.confidence
                },
                'knowledge': {
                    'documents': [doc.dict() for doc in knowledge_result.documents],
                    'answer': knowledge_result.answer,
                    'confidence': knowledge_result.confidence
                }
            }

            context = {
                'user_query': user_message,
                'user_context': {
                    'user_id': user_id,
                    'user_role': 'user'
                },
                'conversation_history': [],
                'system_capabilities': ['knowledge_retrieval', 'intent_classification']
            }

            final_response = await self.response_generator.generate_response(
                execution_results=execution_results,
                context=context
            )

            results["final_response"] = final_response
            results["processing_complete"] = True

            logger.info("✅ Обработка запроса завершена")

        except Exception as e:
            logger.error(f"❌ Ошибка обработки запроса: {e}")
            results["error"] = str(e)
            results["processing_complete"] = False
            # Резервный ответ при ошибке
            results["final_response"] = (
                "Извините, произошла техническая ошибка. "
                "Пожалуйста, повторите запрос или обратитесь в поддержку по телефону."
            )

        return results

    def print_processing_summary(self, results: Dict[str, Any]):
        """Выводит красивый отчет о процессе обработки"""
        print("\n" + "=" * 60)
        print("📊 ОТЧЕТ ОБ ОБРАБОТКЕ ЗАПРОСА")
        print("=" * 60)

        print(f"👤 Пользователь: {results['user_id']}")
        print(f"💬 Сообщение: {results['user_message']}")
        print()

        steps = results["processing_steps"]

        # Токсичность
        toxicity = steps.get("toxicity_check", {})
        print(f"🔍 ТОКСИЧНОСТЬ: {toxicity.get('status', 'N/A')}")
        print(f"   Оценка: {toxicity.get('score', 0):.3f}")
        if toxicity.get('reasons'):
            print(f"   Причины: {', '.join(toxicity['reasons'][:2])}")

        # Деэскалация
        deescalation = steps.get("deescalation", {})
        if deescalation.get('applied'):
            print(f"🔄 ДЕЭСКАЛАЦИЯ: ПРИМЕНЕНА")
            print(f"   Ответ: {deescalation.get('deescalated_message', 'N/A')}")

        # Намерение
        intent = steps.get("intent_classification", {})
        print(f"🎯 НАМЕРЕНИЕ: {intent.get('intent', 'N/A')}")
        print(f"   Уверенность: {intent.get('confidence', 0):.2f}")

        # База знаний
        knowledge = steps.get("knowledge_retrieval", {})
        print(f"📚 БАЗА ЗНАНИЙ: {knowledge.get('documents_found', 0)} документов")
        print(f"   Уверенность: {knowledge.get('confidence', 0):.2f}")

        print("\n" + "🤖 ФИНАЛЬНЫЙ ОТВЕТ:")
        print("-" * 40)
        print(results["final_response"])
        print("=" * 60)


async def main():
    """Главная функция для демонстрации работы системы"""

    # Инициализация системы
    system = TechSupportSystem()
    success = await system.initialize()

    if not success:
        print("❌ Не удалось инициализировать систему техподдержки")
        return

    print("🚀 Система техподдержки запущена!")
    print("Введите сообщения для обработки (или 'exit' для выхода)")
    print("-" * 50)

    # Тестовые запросы для демонстрации
    test_messages = [
        "Здравствуйте, не могу войти в систему, выдает ошибку аутентификации",
        "Вы все идиоты! Почему система постоянно падает?!",
        "Как мне сбросить пароль от учетной записи?",
        "Ваша поддержка - просто шутка! Никто не может нормально помочь!",
        "Нужна информация о текущих тарифных планах",
        "Система работает очень медленно, что делать?"
    ]

    # Обработка тестовых сообщений
    for i, message in enumerate(test_messages, 1):
        print(f"\n📨 Тестовый запрос {i}/{len(test_messages)}:")
        print(f"💬 '{message}'")

        results = await system.process_user_request(message, f"test_user_{i}")
        system.print_processing_summary(results)

        # Пауза между запросами для удобства чтения
        if i < len(test_messages):
            input("\n⏎ Нажмите Enter для следующего запроса...")

    # Интерактивный режим
    print("\n🎮 ПЕРЕХОД В ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Вводите свои сообщения для обработки:")

    while True:
        try:
            user_input = input("\n💬 Ваше сообщение: ").strip()

            if user_input.lower() in ['exit', 'выход', 'quit']:
                print("👋 До свидания!")
                break

            if not user_input:
                print("⚠️ Пожалуйста, введите сообщение")
                continue

            results = await system.process_user_request(user_input, "interactive_user")
            system.print_processing_summary(results)

        except KeyboardInterrupt:
            print("\n👋 Завершение работы...")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
