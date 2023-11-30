from abc import abstractmethod


class Task:
    @abstractmethod
    def execute(self):
        pass
