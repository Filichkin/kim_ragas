# RAGAS Testset Generator

Генератор тестовых датасетов для оценки RAG (Retrieval-Augmented Generation) систем с поддержкой русского языка на базе библиотеки [Ragas](https://github.com/explodinggradients/ragas).

## 🚀 Возможности

- **Генерация вопросов и ответов на русском языке** с использованием адаптированных промптов
- **Single-hop синтезатор** для создания конкретных вопросов по документам
- **Автоматическая загрузка PDF документов** из указанной директории
- **Современное логирование** с использованием loguru
- **Модульная архитектура** с правильной структурой Python пакетов
- **Интеграция с OpenAI** для генерации контента

## 📁 Структура проекта

```
kim_ragas/
├── __init__.py                 # Корневой пакет
├── config.py                   # Конфигурация приложения
├── ragas_eval/                 # Пакет для оценки RAG систем
│   ├── __init__.py            # Инициализация пакета
│   ├── script.py              # Основной скрипт генерации
│   └── patches/               # Патчи для Ragas
│       ├── __init__.py
│       ├── question_potential_patch.py
│       └── themes_patch.py
├── data/                      # Директория с PDF документами
│   └── service_book.pdf
├── draft/                     # Черновики документов
├── requirements.txt           # Зависимости Python
└── README.md                 # Документация
```

## 🛠️ Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/Filichkin/kim_ragas.git
cd kim_ragas
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка конфигурации

Создайте файл `.env` в корне проекта:

```env
OPENAI_API_KEY=your_openai_api_key_here
DATA_DIR=./data
```

## 🚀 Быстрый старт

### Запуск генерации тестового датасета

```bash
python -m ragas_eval.script
```

Скрипт автоматически:
1. Загрузит PDF документы из директории `data/`
2. Создаст синтезатор с русскими промптами
3. Сгенерирует 10 вопросов и ответов
4. Сохранит результат в `ragas_testset_singlehop_ru.csv`

### Пример вывода

```
2025-09-16 14:40:21 | INFO     | Начинаем генерацию тестового датасета на русском языке
2025-09-16 14:40:21 | INFO     | Загружаем PDF документы из data
2025-09-16 14:40:22 | INFO     | Загружено 6 документов
2025-09-16 14:40:22 | INFO     | Инициализируем LLM и эмбеддинги
2025-09-16 14:40:22 | INFO     | Создаем Single-hop синтезатор
2025-09-16 14:40:22 | INFO     | Адаптируем промпты для русского языка
2025-09-16 14:40:29 | INFO     | Промпты успешно адаптированы для русского языка
2025-09-16 14:40:29 | INFO     | Генерируем тестовый набор из документов
2025-09-16 14:40:35 | SUCCESS  | Готово! Записей: 10. Файл: ragas_testset_singlehop_ru.csv
```

## 📊 Формат выходных данных

Генерируемый CSV файл содержит следующие колонки:

- `user_input` - Сгенерированный вопрос на русском языке
- `reference_contexts` - Контекст из документа для ответа
- `reference` - Эталонный ответ на русском языке
- `synthesizer_name` - Название использованного синтезатора

### Пример данных

```csv
user_input,reference_contexts,reference,synthesizer_name
"Какие обязательства берет на себя владелец автомобиля Kia согласно предоставленной информации?","['Сервисная книжка Добро пожаловать в Семью Владельцев автомобилей Kia...']","Владелец автомобиля Kia берет на себя обязательства соблюдать условия, изложенные в Руководстве по эксплуатации автомобиля и в сервисной книжке...",single_hop_specific_query_synthesizer
```

## ⚙️ Конфигурация

### Основные параметры в `config.py`

```python
class Config(BaseSettings):
    # API конфигурация
    OPENAI_API_KEY: str
    OPENROUTER_API_KEY: str
    
    # Пути к данным
    DATA_DIR: str = "./data"
    
    # Настройки модели
    LLM_AGENT_MODEL: str = "openai/gpt-oss-20b:free"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000
```

### Настройка в скрипте

В `ragas_eval/script.py` можно изменить:

```python
# Размер тестового набора
testset_size=10

# Модель для генерации
model='gpt-3.5-turbo'  # или 'gpt-4o-mini'

# Выходной файл
output_file = 'ragas_testset_singlehop_ru.csv'
```

## 🔧 Технические детали

### Используемые технологии

- **Ragas 0.3.4** - библиотека для оценки RAG систем
- **LangChain** - загрузка и обработка документов
- **OpenAI GPT** - генерация вопросов и ответов
- **Loguru** - современное логирование
- **Pydantic** - валидация конфигурации

### Архитектура

1. **Загрузка документов**: PDF файлы загружаются через `PDFPlumberLoader`
2. **Инициализация LLM**: Настройка OpenAI модели для генерации
3. **Создание синтезатора**: `SingleHopSpecificQuerySynthesizer` для конкретных вопросов
4. **Адаптация промптов**: Автоматическая русификация промптов Ragas
5. **Генерация**: Создание вопросов и ответов на русском языке
6. **Сохранение**: Экспорт в CSV формат

### Патчи Ragas

Проект включает патчи для исправления проблем совместимости:

- `question_potential_patch.py` - исправление валидации score
- `themes_patch.py` - исправление обработки кортежей в themes

## 📝 Примеры использования

### Программное использование

```python
import asyncio
from ragas_eval.script import main

# Запуск генерации
asyncio.run(main())
```

### Изменение параметров

```python
# В ragas_eval/script.py
async def main():
    # Изменить размер тестового набора
    dataset = generator.generate_with_langchain_docs(
        docs,
        testset_size=20,  # Увеличить до 20 записей
        query_distribution=query_distribution,
    )
```

## 🐛 Устранение неполадок

### Проблема: "OPENAI_API_KEY не задан"

**Решение**: Убедитесь, что в файле `.env` указан корректный API ключ OpenAI.

### Проблема: "Папка с данными не найдена"

**Решение**: Создайте директорию `data/` и поместите в неё PDF файлы.

### Проблема: "В data не найдено PDF-файлов"

**Решение**: Убедитесь, что в директории `data/` есть файлы с расширением `.pdf`.

### Проблема: Ошибки импорта

**Решение**: Запускайте скрипт из корневой директории проекта:
```bash
python -m ragas_eval.script
```

## 📋 Требования

- **Python 3.8+**
- **OpenAI API ключ**
- **PDF документы** для обработки

### Основные зависимости

- `ragas==0.3.4`
- `langchain-community`
- `langchain-openai`
- `loguru`
- `pydantic-settings`
- `openai`

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте изменения (`git commit -m 'Add amazing feature'`)
4. Отправьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

## 🙏 Благодарности

- [Ragas](https://github.com/explodinggradients/ragas) - за отличную библиотеку для оценки RAG систем
- [LangChain](https://github.com/langchain-ai/langchain) - за инструменты для работы с документами
- [OpenAI](https://openai.com/) - за API для генерации контента

## 📞 Поддержка

Если у вас есть вопросы или проблемы, создайте [Issue](https://github.com/Filichkin/kim_ragas/issues) в репозитории.

---

**Создано с ❤️ для русскоязычного сообщества RAG разработчиков**