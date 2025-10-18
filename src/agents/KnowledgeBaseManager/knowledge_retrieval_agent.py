import logging
from typing import List, Dict, Any
import re

from src.agents.interface_agents.knowledge_retrieval_agent import (
    AbstractKnowledgeRetrievalAgent, KnowledgeRetrievalResult, RetrievedDocument
)
from src.agents.KnowledgeBaseManager.knowledge_base_manager import KnowledgeBaseManager

logger = logging.getLogger(__name__)


class KnowledgeRetrievalAgent(AbstractKnowledgeRetrievalAgent):
    """Реализация агента для извлечения знаний из базы"""

    def __init__(self, knowledge_base_manager: KnowledgeBaseManager, similarity_threshold: float = 0.7):
        """
        Args:
            knowledge_base_manager: Менеджер базы знаний
            similarity_threshold: Порог схожести для релевантности
        """
        self.knowledge_base_manager = knowledge_base_manager
        self.similarity_threshold = similarity_threshold
        self._response_templates = self._load_response_templates()

    def _load_response_templates(self) -> Dict[str, str]:
        """Загружает шаблоны ответов"""
        return {
            "high_confidence": "На основе информации из базы знаний: {answer}",
            "medium_confidence": "Согласно имеющейся информации: {answer}",
            "low_confidence": "Похожая информация из базы знаний: {answer}",
            "no_results": "К сожалению, в базе знаний нет информации по вашему вопросу. Рекомендую уточнить запрос или обратиться к специалисту."
        }

    async def retrieve_knowledge(self, query: str, context: Dict[str, Any]) -> KnowledgeRetrievalResult:
        """
        Извлекает релевантную информацию из базы знаний

        Args:
            query: Поисковый запрос
            context: Контекст поиска (фильтры, настройки)

        Returns:
            KnowledgeRetrievalResult: Результат извлечения знаний
        """
        try:
            # Извлекаем документы из базы знаний
            documents_with_scores = await self.knowledge_base_manager.similarity_search_with_score(
                query=query,
                k=context.get('k', 5),
                filters=context.get('filters')
            )

            # Фильтруем по порогу схожести
            filtered_documents = [
                (doc, score) for doc, score in documents_with_scores
                if score >= self.similarity_threshold
            ]

            # Преобразуем в формат RetrievedDocument
            retrieved_docs = []
            for doc, score in filtered_documents:
                retrieved_docs.append(RetrievedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    similarity_score=float(score),
                    source=doc.metadata.get('source', 'unknown')
                ))

            # Генерируем ответ на основе найденных документов
            answer, confidence = await self._generate_answer_from_documents(query, retrieved_docs)

            # Генерируем предлагаемые действия и вопросы
            suggested_actions = self._generate_suggested_actions(retrieved_docs, query)
            additional_questions = self._generate_additional_questions(retrieved_docs, query)

            return KnowledgeRetrievalResult(
                documents=retrieved_docs,
                answer=answer,
                confidence=confidence,
                suggested_actions=suggested_actions,
                additional_questions=additional_questions
            )

        except Exception as e:
            logger.error(f"Ошибка при извлечении знаний: {e}")
            return KnowledgeRetrievalResult(
                documents=[],
                answer=self._response_templates["no_results"],
                confidence=0.0,
                suggested_actions=["Обратиться к специалисту поддержки"],
                additional_questions=[]
            )

    async def _generate_answer_from_documents(self, query: str, documents: List[RetrievedDocument]) -> tuple[
        str, float]:
        """
        Генерирует ответ на основе найденных документов

        Args:
            query: Исходный запрос
            documents: Найденные документы

        Returns:
            tuple: (ответ, уверенность)
        """
        if not documents:
            return self._response_templates["no_results"], 0.0

        # Вычисляем общую уверенность
        total_confidence = sum(doc.similarity_score for doc in documents) / len(documents)

        # Извлекаем ключевую информацию из документов
        key_information = self._extract_key_information(documents, query)

        # Выбираем шаблон ответа на основе уверенности
        if total_confidence > 0.8:
            template = self._response_templates["high_confidence"]
        elif total_confidence > 0.6:
            template = self._response_templates["medium_confidence"]
        else:
            template = self._response_templates["low_confidence"]

        answer = template.format(answer=key_information)
        return answer, total_confidence

    def _extract_key_information(self, documents: List[RetrievedDocument], query: str) -> str:
        """
        Извлекает ключевую информацию из документов

        Args:
            documents: Найденные документы
            query: Исходный запрос

        Returns:
            str: Ключевая информация
        """
        if not documents:
            return "Информация не найдена"

        # Сортируем документы по релевантности
        sorted_docs = sorted(documents, key=lambda x: x.similarity_score, reverse=True)

        # Берем наиболее релевантные части из топ-3 документов
        key_parts = []
        for doc in sorted_docs[:3]:
            # Извлекаем наиболее релевантные предложения
            sentences = self._extract_relevant_sentences(doc.content, query)
            if sentences:
                key_parts.extend(sentences[:2])  # Берем до 2 предложений из каждого документа

        # Убираем дубликаты и объединяем
        unique_parts = list(dict.fromkeys(key_parts))
        return " ".join(unique_parts) if unique_parts else sorted_docs[0].content[:200] + "..."

    def _extract_relevant_sentences(self, text: str, query: str) -> List[str]:
        """
        Извлекает релевантные предложения из текста

        Args:
            text: Исходный текст
            query: Поисковый запрос

        Returns:
            List[str]: Релевантные предложения
        """
        # Разбиваем текст на предложения
        sentences = re.split(r'[.!?]+', text)

        # Ключевые слова из запроса
        query_keywords = set(query.lower().split())

        # Находим предложения, содержащие ключевые слова
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Проверяем совпадение ключевых слов
            keyword_matches = sum(1 for keyword in query_keywords if keyword in sentence_lower)
            if keyword_matches > 0 and len(sentence.strip()) > 10:
                relevant_sentences.append(sentence.strip())

        return relevant_sentences

    def _generate_suggested_actions(self, documents: List[RetrievedDocument], query: str) -> List[str]:
        """
        Генерирует предлагаемые действия на основе найденной информации

        Args:
            documents: Найденные документы
            query: Исходный запрос

        Returns:
            List[str]: Предлагаемые действия
        """
        actions = []

        if not documents:
            actions.append("Уточните формулировку запроса")
            actions.append("Обратитесь к специалисту поддержки")
            return actions

        # Анализируем содержание документов для предложения действий
        all_content = " ".join([doc.content for doc in documents]).lower()

        # Предлагаем действия на основе тематики
        if any(word in all_content for word in ["оформлен", "заявк", "документ"]):
            actions.append("Оформить официальную заявку")

        if any(word in all_content for word in ["контакт", "телефон", "email"]):
            actions.append("Связаться с ответственным сотрудником")

        if any(word in all_content for word in ["инструкц", "руководство", "мануал"]):
            actions.append("Ознакомиться с полной инструкцией")

        if any(word in all_content for word in ["срок", "время", "период"]):
            actions.append("Уточнить сроки выполнения")

        # Общие действия
        actions.extend([
            "Сохранить информацию для дальнейшего использования",
            "Поделиться информацией с заинтересованными лицами"
        ])

        return actions[:5]  # Ограничиваем количество действий

    def _generate_additional_questions(self, documents: List[RetrievedDocument], query: str) -> List[str]:
        """
        Генерирует дополнительные вопросы для уточнения

        Args:
            documents: Найденные документы
            query: Исходный запрос

        Returns:
            List[str]: Дополнительные вопросы
        """
        questions = []

        if not documents or len(documents) == 0:
            questions.append("Можете уточнить, что именно вас интересует?")
            questions.append("Есть ли конкретные аспекты, которые вас волнуют?")
            return questions

        # Анализируем тематику для генерации уточняющих вопросов
        all_content = " ".join([doc.content for doc in documents]).lower()

        if any(word in all_content for word in ["стипенд", "выплат", "деньги"]):
            questions.extend([
                "Вас интересуют условия получения стипендии?",
                "Нужна информация о сроках выплат?",
                "Интересует размер стипендии?"
            ])

        if any(word in all_content for word in ["программ", "курс", "обучен"]):
            questions.extend([
                "Вас интересуют конкретные программы обучения?",
                "Нужна информация о требованиях к поступлению?",
                "Интересует расписание занятий?"
            ])

        if any(word in all_content for word in ["требован", "услов", "критери"]):
            questions.extend([
                "Уточните, какие именно требования вас интересуют?",
                "Нужна информация о критериях отбора?",
                "Интересуют особые условия?"
            ])

        # Общие уточняющие вопросы
        if len(questions) < 3:
            questions.extend([
                "Уточните временные рамки вашего вопроса?",
                "Есть ли дополнительные детали, которые стоит учесть?",
                "Интересует ли вас практическое применение этой информации?"
            ])

        return questions[:3]  # Ограничиваем количество вопросов

    async def search_with_filters(self, query: str, filters: Dict[str, Any], k: int = 5) -> KnowledgeRetrievalResult:
        """
        Поиск с дополнительными фильтрами

        Args:
            query: Поисковый запрос
            filters: Фильтры для поиска
            k: Количество результатов

        Returns:
            KnowledgeRetrievalResult: Результат поиска
        """
        context = {
            'k': k,
            'filters': filters
        }

        return await self.retrieve_knowledge(query, context)

    async def get_related_topics(self, query: str) -> List[str]:
        """
        Возвращает связанные темы для запроса

        Args:
            query: Поисковый запрос

        Returns:
            List[str]: Список связанных тем
        """
        try:
            # Ищем похожие документы
            documents = await self.knowledge_base_manager.similarity_search(
                query=query,
                k=10
            )

            # Извлекаем темы из метаданных и содержания
            topics = set()

            for doc in documents:
                # Из метаданных
                if 'Header1' in doc.metadata:
                    topics.add(doc.metadata['Header1'])
                if 'Header2' in doc.metadata:
                    topics.add(doc.metadata['Header2'])

                # Из содержания (первые слова)
                first_words = ' '.join(doc.page_content.split()[:5])
                if len(first_words) > 10:
                    topics.add(first_words)

            return list(topics)[:8]  # Ограничиваем количество тем

        except Exception as e:
            logger.error(f"Ошибка при получении связанных тем: {e}")
            return []

    async def get_knowledge_base_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о базе знаний

        Returns:
            Dict: Информация о базе знаний
        """
        return await self.knowledge_base_manager.get_knowledge_base_stats()

    def set_similarity_threshold(self, threshold: float):
        """
        Устанавливает порог схожести

        Args:
            threshold: Новый порог схожести (0.0 - 1.0)
        """
        if 0 <= threshold <= 1:
            self.similarity_threshold = threshold
            logger.info(f"Порог схожести обновлен: {threshold}")
        else:
            raise ValueError("Порог схожести должен быть между 0 и 1")


