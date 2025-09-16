import logging

from loguru import logger


def setup_logger(
    level: str = 'INFO',
    colorize: bool = True,
    format_string: str = None,
    enable_file_logging: bool = False,
    log_file: str = 'ragas_eval.log'
) -> None:
    """
    Настройка логгера loguru с указанными параметрами.

    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        colorize: Включить цветное форматирование
        format_string: Кастомная строка форматирования
        enable_file_logging: Включить логирование в файл
        log_file: Путь к файлу логов
    """
    # Убираем стандартный обработчик
    logger.remove()

    # Стандартная строка форматирования
    if format_string is None:
        format_string = (
            '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
            '<level>{level: <8}</level> | '
            '<cyan>{name}</cyan>:<cyan>{function}</cyan>:'
            '<cyan>{line}</cyan> - <level>{message}</level>'
        )

    # Настройка консольного вывода
    logger.add(
        lambda msg: print(msg, end=''),
        format=format_string,
        level=level,
        colorize=colorize,
    )

    # Настройка файлового логирования (опционально)
    if enable_file_logging:
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation='10 MB',
            retention='7 days',
            compression='zip',
        )

    # Приглушаем шумные предупреждения парсера PDF
    logging.getLogger('pdfminer').setLevel(logging.ERROR)

    logger.info(f'Логгер настроен с уровнем {level}')


def get_logger():
    """
    Получить настроенный экземпляр логгера.

    Returns:
        Настроенный экземпляр loguru logger
    """
    return logger


# Предустановленные конфигурации
def setup_development_logger():
    """Настройка логгера для разработки."""
    setup_logger(
        level='DEBUG',
        colorize=True,
        enable_file_logging=True,
        log_file='logs/development.log'
    )


def setup_production_logger():
    """Настройка логгера для продакшена."""
    setup_logger(
        level='INFO',
        colorize=False,
        enable_file_logging=True,
        log_file='logs/production.log'
    )


def setup_simple_logger():
    """Простая настройка логгера для быстрого старта."""
    setup_logger(
        level='INFO',
        colorize=True,
        enable_file_logging=False
    )
