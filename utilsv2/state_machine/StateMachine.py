
from enum import Enum, auto


class State(Enum):
    UNDEFINE = -1
    VACANT = 1
    AVAILABLE = 2
    OCCUPIED = 3

class TB_StateMachine:
    def __init__(self, initial_state=State.UNDEFINE):
        if initial_state in [State.VACANT, State.AVAILABLE, State.OCCUPIED, State.UNDEFINE]:
            self.state = initial_state
        else:
            raise ValueError("Invalid initial state")

    def get_state(self):
        return self.state

    def to_available(self):
        if self.state == State.VACANT:
            self.state = State.AVAILABLE
            return 0
        return -1  # 狀態轉換失敗

    def to_occupied(self):
        if self.state == State.AVAILABLE:
            self.state = State.OCCUPIED
            return 0
        return -1  # 狀態轉換失敗

    def to_vacant(self):
        if self.state == State.OCCUPIED:
            self.state = State.VACANT
            return 0
        return -1  # 狀態轉換失敗


if __name__ == "__main__":
    #
    state_machine = TB_StateMachine()
    print(state_machine.get_state())  # 初始狀態 State.VACANT
    state_machine.to_available()  # 轉換狀態到 State.AVAILABLE
    print(state_machine.get_state())
    state_machine.to_occupied()  # 轉換狀態到 State.OCCUPIED
    print(state_machine.get_state())
    state_machine.to_vacant()  # 轉換狀態到 State.VACANT
    print(state_machine.get_state())