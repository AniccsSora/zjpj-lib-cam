
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

    def to_undefine(self):
        self.state = State.UNDEFINE
        return 0

    def get_truth_state_v2(self, n, queue: Queue, use_last=False):
        """
        檢查 queue 中最後 n 個元素是否相同
        :param n: 檢查的最後 n 個元素，連續 n 個元素都相同的話 回傳 該狀態。
        :return: True 如果最後 n 個元素相同，False 否則
        """
        ll = queue._get_queue_lst().copy()

        # 如果 queue 長度小於 n，無法進行判斷
        if len(ll) < n:
            # 如果列表為空
            print(" [狀態判斷錯誤] list < n ({})。".format(n))
            return -1
        element_count = Counter(ll)
        # 檢查最後 n 個元素是否相同
        if use_last:
            last_n_elements = ll[-n:]  # 取得最後 n 個元素
        else:
            last_n_elements = ll[:n]  # 取得前面 n 個元素
        print("ALL :", ll)
        if use_last:
            print(f"last {n}:", last_n_elements)
        else:
            print(f"first {n}:", last_n_elements)

        # ll[0].value  >> 1
        # ll[0].name >> 'VACANT'

        if len(set(last_n_elements)) == 1:  # 判斷集合長度是否為 1，表示元素相同)
            if use_last:
                return ll[-1].name  # 直接回傳 連續 n 個狀態相同的狀態
            else:
                return ll[0].name
        else:
            # return 'VACANT'
            # 移除 undefine
            while (1):
                try:
                    ll.remove(State.UNDEFINE)
                except ValueError:
                    #  not in list Error raised.
                    break
            element_count = Counter(ll)
            if len(element_count):
                # most_common(1) return 最多的元素
                most_cnt = element_count.most_common(1)[0][0]
                return most_cnt.name
            else:
                return 'VACANT'

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
                #  not in list Error raised.
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