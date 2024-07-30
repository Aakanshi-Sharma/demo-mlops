import pytest

class NotInRange(Exception):
    def __init__(self, message="Not in Range"):
        self.message=message
        super().__init__(self.message)


def testing():
    a=5
    with pytest.raises(NotInRange):
        if a not in range(10,15):
            raise NotInRange