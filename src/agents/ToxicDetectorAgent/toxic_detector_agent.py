import asyncio
from typing import List
import logging
from detoxify import Detoxify

from src.agents.interface_agents.agent_response import SafetyCheckResult, SafetyStatus
from src.agents.interface_agents.toxic_detector_agent_interface import AbstractToxicityDetectorAgent

logger = logging.getLogger(__name__)


class ToxicityDetectorAgent(AbstractToxicityDetectorAgent):
    """Реализация агента для детекции токсичности с использованием Detoxify"""

    def __init__(self, toxicity_threshold: float = 0.7, high_toxicity_threshold: float = 0.9):
        """
        Args:
            toxicity_threshold: Порог для определения токсичности (0.0 - 1.0)
            high_toxicity_threshold: Порог для определения высокой токсичности
        """
        self.toxicity_threshold = toxicity_threshold
        self.high_toxicity_threshold = high_toxicity_threshold
        self._model = None
        self._model_lock = asyncio.Lock()

    async def _load_model(self):
        """Ленивая загрузка модели Detoxify"""
        if self._model is None:
            async with self._model_lock:
                if self._model is None:
                    try:
                        logger.info("Loading Detoxify model...")
                        self._model = Detoxify('multilingual')
                        logger.info("Detoxify model loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load Detoxify model: {e}")
                        raise

    async def detect_toxicity(self, text: str) -> SafetyCheckResult:
        """
        Анализирует текст на наличие токсичности и оскорбительного контента

        Args:
            text: Текст для анализа

        Returns:
            SafetyCheckResult: Результат проверки безопасности
        """
        try:
            await self._load_model()

            if not text or not text.strip():
                return SafetyCheckResult(
                    status=SafetyStatus.SAFE,
                    score=0.0,
                    reasons=["Empty text"]
                )

            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None,
                self._model.predict,
                text
            )

            return self._analyze_predictions(predictions, text)

        except Exception as e:
            logger.error(f"Error in toxicity detection: {e}")
            return SafetyCheckResult(
                status=SafetyStatus.UNCERTAIN,
                score=0.0,
                reasons=[f"Detection error: {str(e)}"]
            )

    def _analyze_predictions(self, predictions: dict, text: str) -> SafetyCheckResult:
        """
        Анализирует предсказания модели и формирует результат

        Args:
            predictions: Словарь с предсказаниями от Detoxify
            text: Исходный текст

        Returns:
            SafetyCheckResult: Структурированный результат анализа
        """
        toxic_categories = []
        max_score = 0.0

        for category, score in predictions.items():
            if score > self.toxicity_threshold:
                toxic_categories.append(f"{category}: {score:.3f}")
                max_score = max(max_score, score)

        if max_score > self.high_toxicity_threshold:
            status = SafetyStatus.TOXIC
        elif max_score > self.toxicity_threshold:
            status = SafetyStatus.UNCERTAIN
        else:
            status = SafetyStatus.SAFE

        reasons = self._get_additional_reasons(text, predictions, status)
        if toxic_categories:
            reasons.extend(toxic_categories)

        return SafetyCheckResult(
            status=status,
            score=max_score,
            reasons=reasons
        )

    def _get_additional_reasons(self, text: str, predictions: dict, status: SafetyStatus) -> List[str]:
        """
        Добавляет дополнительные контекстные причины на основе анализа текста

        Args:
            text: Анализируемый текст
            predictions: Предсказания модели
            status: Определенный статус безопасности

        Returns:
            List[str]: Список дополнительных причин
        """
        reasons = []
        text_lower = text.lower()

        offensive_words = ['дурак', 'идиот', 'дебил', 'мудак', 'сволочь', 'подонок', 'тварь']
        found_offensive = [word for word in offensive_words if word in text_lower]
        if found_offensive:
            reasons.append(f"Обнаружены оскорбительные слова: {', '.join(found_offensive)}")

        if len(text) > 10 and sum(1 for c in text if c.isupper()) / len(text) > 0.7:
            reasons.append("Текст написан в основном в верхнем регистре (кричащий стиль)")

        exclamation_count = text.count('!')
        if exclamation_count > 3:
            reasons.append(f"Много восклицательных знаков ({exclamation_count}), возможна агрессивная подача")

        if predictions.get('threat', 0) > 0.3:
            threat_words = ['убью', 'уничтожу', 'разорву', 'сожгу', 'покалечу']
            if any(word in text_lower for word in threat_words):
                reasons.append("Обнаружены возможные угрозы")

        return reasons

    async def get_detailed_toxicity_analysis(self, text: str) -> dict:
        """
        Расширенный анализ токсичности с детализацией по категориям

        Args:
            text: Текст для анализа

        Returns:
            dict: Детализированные результаты анализа
        """
        try:
            await self._load_model()

            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None,
                self._model.predict,
                text
            )

            category_analysis = {}
            for category, score in predictions.items():
                if score > self.high_toxicity_threshold:
                    level = "HIGH"
                elif score > self.toxicity_threshold:
                    level = "MEDIUM"
                else:
                    level = "LOW"

                category_analysis[category] = {
                    'score': float(score),
                    'level': level,
                    'exceeds_threshold': score > self.toxicity_threshold
                }

            overall_result = self._analyze_predictions(predictions, text)

            return {
                'overall': {
                    'status': overall_result.status.value,
                    'max_score': overall_result.score,
                    'reasons': overall_result.reasons
                },
                'categories': category_analysis,
                'thresholds': {
                    'toxicity_threshold': self.toxicity_threshold,
                    'high_toxicity_threshold': self.high_toxicity_threshold
                }
            }

        except Exception as e:
            logger.error(f"Error in detailed toxicity analysis: {e}")
            return {
                'error': str(e),
                'overall': {
                    'status': SafetyStatus.UNCERTAIN.value,
                    'max_score': 0.0,
                    'reasons': [f"Analysis error: {str(e)}"]
                },
                'categories': {}
            }

    def update_thresholds(self, toxicity_threshold: float = None, high_toxicity_threshold: float = None):
        """
        Обновляет пороговые значения для детекции токсичности

        Args:
            toxicity_threshold: Новый порог токсичности
            high_toxicity_threshold: Новый порог высокой токсичности
        """
        if toxicity_threshold is not None:
            if 0 <= toxicity_threshold <= 1:
                self.toxicity_threshold = toxicity_threshold
            else:
                raise ValueError("Toxicity threshold must be between 0 and 1")

        if high_toxicity_threshold is not None:
            if 0 <= high_toxicity_threshold <= 1:
                self.high_toxicity_threshold = high_toxicity_threshold
            else:
                raise ValueError("High toxicity threshold must be between 0 and 1")

        if self.high_toxicity_threshold < self.toxicity_threshold:
            self.high_toxicity_threshold = self.toxicity_threshold


import asyncio
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)


async def main():
    # Создаем экземпляр агента
    toxicity_detector = ToxicityDetectorAgent(
        toxicity_threshold=0.7,
        high_toxicity_threshold=0.9
    )

    # Тестовые примеры
    test_texts = [
        "Здравствуйте, помогите мне пожалуйста с настройкой системы",
        "Вы все идиоты! Ваша система - полное дерьмо! Я вас всех убью!",
        "Система работает медленно, нужно улучшить производительность",
        "ПОЧЕМУ ВСЕ ТАК МЕДЛЕННО РАБОТАЕТ?! ИСПРАВЛЯЙТЕ СВОЙ ХЛАМ!!!",
        "Fuck you!",
        "Да пошёл ты нахуй! Зови оператора",
    ]

    for text in test_texts:
        print(f"\nАнализ текста: '{text}'")

        # Базовый анализ
        result = await toxicity_detector.detect_toxicity(text)
        print(f"Статус: {result.status.value}")
        print(f"Уверенность: {result.score:.3f}")
        print(f"Причины: {result.reasons}")

        # Детализированный анализ
        detailed = await toxicity_detector.get_detailed_toxicity_analysis(text)
        print(f"Детальный анализ: {detailed}")


if __name__ == "__main__":
    asyncio.run(main())
