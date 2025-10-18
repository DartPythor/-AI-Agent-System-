import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
import aiofiles
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter

logging.basicConfig(level=logging.INFO)


class KnowledgeBaseManager:
    """Менеджер для работы с базой знаний на основе FAISS"""

    def __init__(self, data_path: str = "data/data.md", vector_store_path: str = "vector_store"):
        self.data_path = data_path
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.embeddings = None

    async def initialize_embeddings(self):
        """Инициализирует эмбеддинг модель"""
        try:
            self.embeddings = await asyncio.to_thread(
                HuggingFaceEmbeddings,
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_kwargs={"trust_remote_code": True, "device": "cpu"}
            )
            logging.info("Эмбеддинг модель инициализирована")
        except Exception as e:
            logging.error(f"Ошибка инициализации эмбеддинг модели: {e}")
            raise

    async def load_or_create_knowledge_base(self) -> bool:
        """
        Загружает существующую базу знаний или создает новую

        Returns:
            bool: Успешность операции
        """
        try:
            await self.initialize_embeddings()

            # Проверяем существование базы знаний
            if await self._vector_store_exists():
                await self.load_vector_store()
                logging.info("База знаний загружена из файла")
                return True
            else:
                await self.create_knowledge_base()
                logging.info("Новая база знаний создана")
                return True

        except Exception as e:
            logging.error(f"Ошибка загрузки/создания базы знаний: {e}")
            return False

    async def _vector_store_exists(self) -> bool:
        """Проверяет существование векторной базы"""
        required_files = ["index.faiss", "index.pkl"]
        return all(os.path.exists(os.path.join(self.vector_store_path, f)) for f in required_files)

    async def load_vector_store(self):
        """Загружает векторное хранилище"""
        try:
            self.vector_store = await asyncio.to_thread(
                FAISS.load_local,
                folder_path=self.vector_store_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logging.error(f"Ошибка загрузки векторного хранилища: {e}")
            raise

    async def create_knowledge_base(self):
        """Создает новую базу знаний из markdown файлов"""
        try:
            # Генерируем документы из markdown
            documents = await self._generate_documents_from_markdown()

            # Создаем векторное хранилище
            self.vector_store = await asyncio.to_thread(
                FAISS.from_documents,
                documents=documents,
                embedding=self.embeddings,
            )

            # Сохраняем
            await self.save_vector_store()

        except Exception as e:
            logging.error(f"Ошибка создания базы знаний: {e}")
            raise

    async def _generate_documents_from_markdown(self) -> List[Document]:
        """
        Генерирует документы из markdown файлов

        Returns:
            List[Document]: Список документов
        """
        try:
            async with aiofiles.open(self.data_path, encoding="utf-8") as f:
                markdown_content = await f.read()

            # Определяем заголовки для разделения
            headers_to_split_on = [
                ("#", "Header1"),
                ("##", "Header2"),
                ("###", "Header3")
            ]

            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            chunks = await asyncio.to_thread(splitter.split_text, markdown_content)

            documents_list = []
            for chunk in chunks:
                # Формируем структурированное содержание
                doc_content_parts = []

                # Добавляем заголовки в содержание
                for header_level in ["Header1", "Header2", "Header3"]:
                    if header_level in chunk.metadata:
                        doc_content_parts.append(f"{header_level}: {chunk.metadata[header_level]}")

                doc_content_parts.append(f"Содержание: {chunk.page_content}")

                document = Document(
                    page_content="\n".join(doc_content_parts),
                    metadata={
                        **chunk.metadata,
                        "source": self.data_path,
                        "timestamp": datetime.now().isoformat(),
                        "content_length": len(chunk.page_content)
                    }
                )
                documents_list.append(document)

            logging.info(f"Сгенерировано {len(documents_list)} документов из markdown")
            return documents_list

        except Exception as e:
            logging.error(f"Ошибка генерации документов из markdown: {e}")
            return []

    async def save_vector_store(self):
        """Сохраняет векторное хранилище"""
        try:
            if self.vector_store:
                await asyncio.to_thread(
                    self.vector_store.save_local,
                    folder_path=self.vector_store_path
                )
                logging.info("Векторное хранилище сохранено")
        except Exception as e:
            logging.error(f"Ошибка сохранения векторного хранилища: {e}")

    async def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[
        Document]:
        """
        Выполняет поиск похожих документов

        Args:
            query: Поисковый запрос
            k: Количество результатов
            filters: Фильтры для поиска

        Returns:
            List[Document]: Найденные документы
        """
        try:
            if not self.vector_store:
                await self.load_or_create_knowledge_base()

            results = await asyncio.to_thread(
                self.vector_store.similarity_search,
                query=query,
                k=k,
                filter=filters
            )

            logging.info(f"Найдено {len(results)} документов для запроса: {query}")
            return results

        except Exception as e:
            logging.error(f"Ошибка поиска: {e}")
            return []

    async def similarity_search_with_score(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> \
    List[tuple]:
        """
        Выполняет поиск с оценкой схожести

        Args:
            query: Поисковый запрос
            k: Количество результатов
            filters: Фильтры для поиска

        Returns:
            List[tuple]: Документы с оценками схожести
        """
        try:
            if not self.vector_store:
                await self.load_or_create_knowledge_base()

            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query=query,
                k=k,
                filter=filters
            )

            logging.info(f"Найдено {len(results)} документов с оценками для запроса: {query}")
            return results

        except Exception as e:
            logging.error(f"Ошибка поиска с оценкой: {e}")
            return []

    async def add_documents(self, documents: List[Document]):
        """
        Добавляет новые документы в базу знаний

        Args:
            documents: Список документов для добавления
        """
        try:
            if not self.vector_store:
                await self.load_or_create_knowledge_base()

            await asyncio.to_thread(
                self.vector_store.add_documents,
                documents
            )

            await self.save_vector_store()
            logging.info(f"Добавлено {len(documents)} документов в базу знаний")

        except Exception as e:
            logging.error(f"Ошибка добавления документов: {e}")

    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику базы знаний

        Returns:
            Dict: Статистика базы знаний
        """
        try:
            if not self.vector_store:
                await self.load_or_create_knowledge_base()

            # Получаем информацию о документах
            doc_count = len(self.vector_store.docstore._dict) if hasattr(self.vector_store, 'docstore') else 0

            return {
                "document_count": doc_count,
                "vector_store_path": self.vector_store_path,
                "data_source": self.data_path,
                "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Ошибка получения статистики: {e}")
            return {"error": str(e)}
