from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING

from htn.tasks.types.preconditions import Preconditions
from htn.tasks.types.task import Task

METHOD_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")

if TYPE_CHECKING:
    from htn.tasks.types.compound_task import CompoundTask


class Method:
    """
    Method is a class that represents a method of a compound task.

    It contains preconditions and a list of tasks that
    can be executed in a specific order.
    """

    id: str
    name: str
    parent_task: CompoundTask | None
    preconditions: Preconditions
    tasks: list[Task]

    def __init__(
        self,
        id: str,
        name: str,
        preconditions: Preconditions,
        tasks: list[Task],
    ):
        """
        Initialize a method with preconditions and tasks.

        Args:
            id: Stable identifier for the method in the format
                '<compound_task>.<method>'.
            name: Human-readable method name.
            preconditions: The preconditions for the method
            tasks: The tasks to be executed in the method
        """
        normalized_id = id.strip()
        normalized_name = name.strip()

        if not normalized_id:
            raise ValueError("Method id must be not empty")

        if not normalized_name:
            raise ValueError("Method name must be not empty")

        if not METHOD_ID_PATTERN.fullmatch(normalized_id):
            raise ValueError(
                "Method id must follow the format '<namespace>.<method_name>' "
                "using only lowercase letters, numbers, and underscores"
            )

        self.tasks = [copy.deepcopy(task) for task in tasks]
        for task in self.tasks:
            task.method = self

        self.id = normalized_id
        self.name = normalized_name
        self.preconditions = preconditions
        self.parent_task = None

    def get_task(self, index: int) -> Task:
        """
        Get a task from the method by index.

        Args:
            index: The index of the task to retrieve
        Returns:
            The task at the specified index
        """
        return self.tasks[index]

    def get_tasks(self) -> list[Task]:
        """
        Get the tasks of the method.

        Returns:
            The tasks of the method
        """
        return self.tasks

    def get_preconditions(self) -> Preconditions:
        """
        Get the preconditions of the method.

        Returns:
            The preconditions of the method
        """
        return self.preconditions
