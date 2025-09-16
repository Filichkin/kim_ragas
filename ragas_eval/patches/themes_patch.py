"""Патч для исправления проблемы с кортежами в themes."""


def fix_themes_validation(original_init):
    """Патч для исправления проблемы с кортежами в themes."""
    def patched_init(self, *args, **kwargs):
        # Исправляем themes если они являются кортежами
        if 'themes' in kwargs and isinstance(kwargs['themes'], list):
            fixed_themes = []
            for theme in kwargs['themes']:
                if isinstance(theme, tuple):
                    # Берем первый элемент кортежа как строку
                    fixed_themes.append(
                        str(theme[0]) if theme else ''
                    )
                else:
                    fixed_themes.append(str(theme))
            kwargs['themes'] = fixed_themes
        return original_init(self, *args, **kwargs)
    return patched_init


def apply_themes_patch():
    """Применяет патч для ThemesPersonasInput."""
    try:
        from ragas.testset.synthesizers.multi_hop.specific import (
            ThemesPersonasInput
        )
        ThemesPersonasInput.__init__ = fix_themes_validation(
            ThemesPersonasInput.__init__
        )
    except ImportError:
        pass
