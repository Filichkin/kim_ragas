# RAGAS Testset Generator

Генератор тестовых датасетов для оценки RAG (Retrieval-Augmented Generation) систем с поддержкой русского языка на базе библиотеки [Ragas](https://github.com/explodinggradients/ragas).

## 🚀 Возможности

- **Генерация вопросов и ответов на русском языке** с использованием адаптированных промптов
- **Single-hop синтезатор** для создания конкретных вопросов по документам
- **Автоматическая загрузка PDF документов** из указанной директории
- **Современное логирование** с использованием loguru
- **Модульная архитектура** с правильной структурой Python пакетов
- **Интеграция с OpenAI** для генерации контента
- **Организованное сохранение результатов** в отдельную папку

## 📁 Структура проекта

```
kim_ragas/
├── __init__.py                # Корневой пакет
├── config.py                  # Конфигурация приложения
├── ragas_eval/                # Пакет для оценки RAG систем
│   ├── __init__.py            # Инициализация пакета
│   ├── script.py              # Основной скрипт генерации
│   ├── logger_config.py       # Конфигурация логирования
│   └── patches/               # Патчи для Ragas
│       ├── __init__.py
│       ├── question_potential_patch.py
│       └── themes_patch.py
├── data/                      # Директория с PDF документами
│   └── service_book.pdf
├── output/                    # Директория с результатами генерации
├── draft/                     # Черновики документов
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
2025-09-16 14:40:29 | INFO     | ragas_eval.script:main:101 - Промпты успешно адаптированы для русского языка
2025-09-16 14:40:29 | INFO     | ragas_eval.script:main:109 - Генерируем тестовый набор из документов
2025-09-16 14:40:35 | SUCCESS  | ragas_eval.script:main:129 - Готово! Записей: 10. Файл: output/ragas_testset_singlehop_ru.csv
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
3. **Создание синтезатора**: `SingleHopSpecificQuerySynthesizer` для конкретных вопросов
4. **Адаптация промптов**: Автоматическая русификация промптов Ragas
5. **Генерация**: Создание вопросов и ответов на русском языке
6. **Сохранение**: Экспорт в CSV формат в папку `output/`

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

### Проблема: Папка output не создается

**Решение**: Убедитесь, что у приложения есть права на запись в текущую директорию.

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
