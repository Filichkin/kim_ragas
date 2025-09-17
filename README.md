# RAGAS Testset Generator

Генератор тестовых датасетов для оценки RAG (Retrieval-Augmented Generation) систем с поддержкой русского языка на базе библиотеки [Ragas](https://github.com/explodinggradients/ragas).

## 🚀 Возможности

- **Генерация вопросов и ответов на русском языке** с использованием адаптированных промптов
- **Поддержка разных типов вопросов** - single-hop и multi-hop синтезаторы
- **Knowledge Graph на русском языке** - русские summary и themes
- **Автоматическая загрузка PDF документов** из указанной директории
- **Анализ дублирования вопросов** - автоматическое обнаружение повторений
- **Современное логирование** с использованием loguru
- **Модульная архитектура** с правильной структурой Python пакетов
- **Интеграция с OpenAI** для генерации контента
- **Организованное сохранение результатов** в отдельную папку с timestamp

## 📁 Структура проекта

```
kim_ragas/
├── __init__.py                # Корневой пакет
├── config.py                  # Конфигурация приложения
├── ragas_eval/                # Пакет для оценки RAG систем
│   ├── __init__.py            # Инициализация пакета
│   ├── script.py              # Основной скрипт генерации
│   ├── logger_config.py       # Конфигурация логирования
│   ├── russian_transforms.py  # Русские трансформы для Knowledge Graph
│   └── patches/               # Патчи для Ragas
│       ├── __init__.py
│       ├── question_potential_patch.py
│       └── themes_patch.py
├── data/                      # Директория с PDF документами
│   └── service_book.pdf
├── output/                    # Директория с результатами генерации (с timestamp)
├── draft/                     # Черновики документов
├── knowledge_graph.json       # Граф знаний с русскими summary
├── requirements.txt           # Зависимости Python
├── .gitignore                 # Исключения для Git
└── README.md                  # Документация
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
4. Сохранит результат в `output/ragas_testset_singlehop_ru.csv`

### Пример вывода

```
2025-09-16 14:40:21 | INFO     | ragas_eval.logger_config:setup_logger:63 - Логгер настроен с уровнем INFO
2025-09-16 14:40:21 | INFO     | ragas_eval.script:<module>:39 - Патчи Ragas успешно применены
2025-09-16 14:40:21 | INFO     | ragas_eval.script:main:52 - Начинаем генерацию тестового датасета на русском языке
2025-09-16 14:40:21 | INFO     | ragas_eval.script:main:61 - Загружаем PDF документы из data
2025-09-16 14:40:22 | INFO     | ragas_eval.script:main:74 - Загружено 6 документов
2025-09-16 14:40:22 | INFO     | ragas_eval.script:main:82 - Инициализируем LLM и эмбеддинги
2025-09-16 14:40:22 | INFO     | ragas_eval.script:main:93 - Создаем Single-hop синтезатор
2025-09-16 14:40:22 | INFO     | ragas_eval.script:main:97 - Адаптируем промпты для русского языка
2025-09-17 15:21:36 | INFO     | ragas_eval.script:main:101 - Промпты для SingleHopSpecificQuerySynthesizer адаптированы
2025-09-17 15:21:36 | INFO     | ragas_eval.script:main:101 - Промпты для MultiHopSpecificQuerySynthesizer адаптированы
2025-09-17 15:21:36 | INFO     | ragas_eval.script:main:109 - Генерируем тестовый набор из документов
2025-09-17 15:21:36 | INFO     | ragas_eval.script:main:143 - Дублирующихся вопросов не найдено
2025-09-17 15:21:36 | INFO     | ragas_eval.script:main:147 - Распределение по синтезаторам:
2025-09-17 15:21:36 | INFO     | ragas_eval.script:main:148 -   - single_hop_specific_query_synthesizer: 7 вопросов
2025-09-17 15:21:36 | INFO     | ragas_eval.script:main:148 -   - multi_hop_specific_query_synthesizer: 3 вопроса
2025-09-17 15:21:36 | SUCCESS  | ragas_eval.script:main:151 - Готово! Записей: 10. Файл: output/ragas_testset_singlehop_ru_20250917_153852.csv
2025-09-17 15:21:36 | INFO     | ragas_eval.script:main:178 - Создаем русские трансформы для графа знаний
```

## 📊 Формат выходных данных

### CSV файл с вопросами и ответами

Генерируемый CSV файл содержит следующие колонки:

- `user_input` - Сгенерированный вопрос на русском языке
- `reference_contexts` - Контекст из документа для ответа
- `reference` - Эталонный ответ на русском языке
- `synthesizer_name` - Название использованного синтезатора

### Knowledge Graph (knowledge_graph.json)

Дополнительно создается файл `knowledge_graph.json` с графом знаний, содержащий:

- **Русские summary** - краткое содержание документов на русском языке
- **Русские themes** - основные темы документов на русском языке
- **Эмбеддинги** - векторные представления для поиска сходства
- **Связи между узлами** - отношения между документами и их частями

### Пример данных

**CSV файл:**
```csv
user_input,reference_contexts,reference,synthesizer_name
"Какие обязательства берет на себя владелец автомобиля Kia согласно предоставленной информации?","['Сервисная книжка Добро пожаловать в Семью Владельцев автомобилей Kia...']","Владелец автомобиля Kia берет на себя обязательства соблюдать условия, изложенные в Руководстве по эксплуатации автомобиля и в сервисной книжке...",single_hop_specific_query_synthesizer
"Какие условия гарантии предусмотрены в сервисной книжке автомобилей Kia?","['<1-hop> Сервисная книжка...', '<2-hop> Предпродажное сервисное обслуживание...']","В сервисной книжке автомобилей Kia предусмотрены следующие условия гарантии: основная гарантия на 60 месяцев...",multi_hop_specific_query_synthesizer
```

**Knowledge Graph (фрагмент):**
```json
{
  "nodes": [
    {
      "type": "DOCUMENT",
      "properties": {
        "page_content": "Сервисная книжка Добро пожаловать в Семью Владельцев автомобилей Kia...",
        "summary": "Сервисная книжка Kia содержит информацию о гарантийных обязательствах, техническом обслуживании и условиях эксплуатации автомобилей.",
        "themes": ["гарантия", "техническое обслуживание", "эксплуатация", "дилеры"]
      }
    }
  ]
}
```

## ⚙️ Конфигурация

### Основные параметры в `config.py`

```python
class Config(BaseSettings):
    # API конфигурация
    OPENAI_API_KEY: str
    
    # Модель для генерации
    LLM_MODEL: str = 'gpt-3.5-turbo'
    
    # Пути к данным
    DATA_DIR: str = './data'
    
    # Настройки генерации
    TESTSET_SIZE: int = 10
    OUTPUT_DIR: str = './output'
    OUTPUT_FILENAME: str = 'ragas_testset_singlehop_ru.csv'
```

### Настройка через переменные окружения

Вы можете переопределить любые параметры через файл `.env`:

```env
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
TESTSET_SIZE=20
OUTPUT_DIR=./results
OUTPUT_FILENAME=my_dataset.csv
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
3. **Создание синтезаторов**: 
   - `SingleHopSpecificQuerySynthesizer` для простых вопросов
   - `MultiHopSpecificQuerySynthesizer` для сложных многошаговых вопросов
4. **Адаптация промптов**: Автоматическая русификация промптов Ragas
5. **Генерация**: Создание вопросов и ответов на русском языке
6. **Сохранение**: Экспорт в CSV формат в папку `output/`
7. **Knowledge Graph**: Создание графа знаний с русскими summary и themes

### Модуль логирования

Проект использует отдельный модуль `logger_config.py` для настройки логирования:

```python
from ragas_eval.logger_config import setup_simple_logger, get_logger

# Настройка логгера
setup_simple_logger()
logger = get_logger()
```

Доступные предустановки:
- `setup_simple_logger()` - простая настройка для быстрого старта
- `setup_development_logger()` - настройка для разработки с файловым логированием
- `setup_production_logger()` - настройка для продакшена

### Русские трансформы

Проект включает кастомные трансформы для генерации Knowledge Graph на русском языке:

- `russian_transforms.py` - модуль с русскими промптами для извлечения summary и themes
- `RussianSummaryExtractor` - извлечение краткого содержания на русском языке
- `RussianThemesExtractor` - извлечение тем на русском языке
- `russian_transforms()` - функция для создания набора русских трансформов

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

### Изменение параметров через конфигурацию

```python
# В config.py или .env
TESTSET_SIZE=20  # Увеличить до 20 записей
LLM_MODEL='gpt-4o-mini'  # Использовать более мощную модель
OUTPUT_DIR='./results'  # Изменить папку для результатов
```

### Кастомное логирование

```python
from ragas_eval.logger_config import setup_development_logger

# Настройка для разработки
setup_development_logger()
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


## 📄 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

## 🙏 Благодарности

- [Ragas](https://github.com/explodinggradients/ragas) - за отличную библиотеку для оценки RAG систем
- [LangChain](https://github.com/langchain-ai/langchain) - за инструменты для работы с документами
- [OpenAI](https://openai.com/) - за API для генерации контента

## 📞 Поддержка

Если у вас есть вопросы или проблемы, создайте [Issue](https://github.com/Filichkin/kim_ragas/issues) в репозитории.

---
