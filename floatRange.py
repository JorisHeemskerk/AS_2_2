import inspect

from dataclasses import dataclass
from typing import Annotated, get_type_hints


def check_annotated(func)-> None:
    """
    Checker wrapper function to force type annotations.

    @param func: function to wrap around
    """
    hints = get_type_hints(func, include_extras=True)
    spec = inspect.getfullargspec(func)
    def wrapper(*args, **kwargs):
        arguments = list(args) + list(kwargs.values())
        for idx, arg_name in enumerate(spec[0]):
            hint = hints.get(arg_name)
            validators = getattr(hint, '__metadata__', None)
            if not validators:
                continue
            for validator in validators:
                validator.validate_value(arguments[idx])
        return func(*args, **kwargs)
    return wrapper


@dataclass
class FloatRange:
    """
    FloatRange class

    This class is used to limit the values for floats in certain parameters
    """
    min: float
    max: float

    def validate_value(self, x: float)-> None:
        """
        Validation function for value using `FloatRange`

        provided `x` should be between self.min and self.max (inclusive)

        @param x: float to validate
        """
        if not (self.min <= x <= self.max):
            raise ValueError(f'{x} must be in range [{self.min}, {self.max}].')
