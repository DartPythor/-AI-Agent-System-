from abc import ABC, abstractmethod


class AbstractToxicDetectorAgent(ABC):
    """Абстрактный класс для агентов выполняющий анализ текста на токсичность"""
    @abstractmethod
    async def analys(self, data: str):
        """Определение насколько текст токсичен"""
