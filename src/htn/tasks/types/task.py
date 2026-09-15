from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from htn.tasks.types.compound_task import CompoundTask
    from htn.tasks.types.method import Method


class Task(ABC):
    """
    Base class for all tasks
    It contains the basic methods and fields.
    """

    method: Method | None

    @property
    def parent_task(self) -> CompoundTask | None:
        return self.method.parent_task if self.method else None

    def __init__(self, name: str):
        """
        Initialise the task
        Args:
            name: Name of the task
        """
        self.name = name
        self.method = None

    def __repr__(self) -> str:
        """
        String representation of the task
        Returns:
            String representation of the task
        """
        return f"{self.__class__.__name__}(name={self.name!r})"
