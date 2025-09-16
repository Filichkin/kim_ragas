"""Патч для исправления проблемы с парсингом score."""

import json
from pydantic import ValidationError


def fix_question_potential_validation(original_init):
    """Патч для исправления проблемы с парсингом score."""
    def patched_init(self, *args, **kwargs):
        # Исправляем score если он является объектом
        if 'score' in kwargs and isinstance(kwargs['score'], dict):
            # Извлекаем числовое значение из объекта
            score_obj = kwargs['score']
            if 'description' in score_obj:
                # Пытаемся извлечь числовое значение из описания
                description = score_obj['description'].lower()
                if ('highly relevant' in description or
                        'directly reflects' in description):
                    kwargs['score'] = 5
                elif ('aligns well' in description or
                      'covering the main themes' in description):
                    kwargs['score'] = 4
                elif ('generally reflects' in description or
                      'may miss key details' in description):
                    kwargs['score'] = 3
                elif ('partially aligns' in description or
                      'unrelated details' in description):
                    kwargs['score'] = 2
                elif ('irrelevant' in description or
                      'does not align' in description):
                    kwargs['score'] = 1
                else:
                    kwargs['score'] = 3  # Средний балл по умолчанию
            else:
                kwargs['score'] = 3
        return original_init(self, *args, **kwargs)
    return patched_init


def patch_json_parsing():
    """Патч для исправления ошибок парсинга JSON в RAGAS."""
    try:
        from ragas.testset.transforms.filters import QuestionPotentialOutput

        # Патчим model_validate_json для Pydantic v2
        original_validate_json = (
            QuestionPotentialOutput.model_validate_json
        )

        def patched_validate_json(cls, json_data, **kwargs):
            try:
                return original_validate_json(json_data, **kwargs)
            except (ValidationError, json.JSONDecodeError, Exception) as e:
                print(f"JSON validation failed, using default score: {e}")
                return cls(score=3)

        QuestionPotentialOutput.model_validate_json = classmethod(
            patched_validate_json
        )

        # Также патчим model_validate для общих случаев
        original_validate = QuestionPotentialOutput.model_validate

        def patched_validate(cls, obj, **kwargs):
            try:
                return original_validate(obj, **kwargs)
            except (ValidationError, json.JSONDecodeError, Exception) as e:
                print(f"Model validation failed, using default score: {e}")
                return cls(score=3)

        QuestionPotentialOutput.model_validate = classmethod(
            patched_validate
        )

    except ImportError:
        pass


def apply_question_potential_patch():
    """Применяет патчи для QuestionPotentialOutput."""
    try:
        from ragas.testset.transforms.filters import QuestionPotentialOutput
        QuestionPotentialOutput.__init__ = fix_question_potential_validation(
            QuestionPotentialOutput.__init__
        )
    except ImportError:
        pass

    # Применяем дополнительный патч для JSON парсинга
    patch_json_parsing()
