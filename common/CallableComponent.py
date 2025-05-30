# common/callable_component.py
from abc import abstractmethod
from common.BaseComponent import BaseComponent

class CallableComponent(BaseComponent):
    """
    A component that *must* implement __call__().
    """
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Users can treat this instance like a function.
        """
        raise NotImplementedError