class StateMachine:
    def __init__(self, initial_state='vacant'):
        if initial_state in ['vacant', 'available', 'occupied']:
            self.state = initial_state
        else:
            raise ValueError("Invalid initial state")

    def get_state(self):
        return self.state

    def to_available(self):
        if self.state == 'vacant':
            self.state = 'available'
            return 0
        return -1  # 狀態轉換失敗

    def to_occupied(self):
        if self.state == 'available':
            self.state = 'occupied'
            return 0
        return -1  # 狀態轉換失敗

    def to_vacant(self):
        if self.state == 'occupied':
            self.state = 'vacant'
            return 0
        return -1  # 狀態轉換失敗


if __name__ == "__main__":
    #
    state_machine = StateMachine()
    print(state_machine.get_state())  # 初始狀態 'vacant'
    state_machine.to_available()  # 轉換狀態到 'available'
    print(state_machine.get_state())
    state_machine.to_occupied()  # 轉換狀態到 'occupied'
    print(state_machine.get_state())
    state_machine.to_vacant()  # 轉換狀態到 'vacant'
    print(state_machine.get_state())