
from enum import Enum, auto
from utilsv2.my_queue.qqueue import Queue
from collections import Counter
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

    def get_state_str(self):
        return str(self.state).split('.')[-1]

    def to_available(self):
        self.state = State.AVAILABLE
        return 0

    def to_occupied(self):
        self.state = State.OCCUPIED
        return 0

    def to_vacant(self):
        self.state = State.VACANT
        return 0


    def get_truth_state(self, queue: Queue):
        """
        Base on queue, 來判斷真正的狀態
        # 藉由過去的質狀態積累，來判斷真正的狀態。
        :return:
        """
        ll = queue._get_queue_lst().copy()
        # 移除 undefine
        while (1):
            try:
                ll.remove(State.UNDEFINE)
            except ValueError:
                break
        element_count = Counter(ll)

        # most_common(1) return 最多的元素
        if element_count:
            # 返回最多次數元素的 enum's str.
            most_cnt = element_count.most_common(1)[0][0]

            return str(most_cnt).split('.')[-1]
        else:
            # 如果列表為空
            print(" [狀態判斷錯誤] list為空。")
            return -1

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