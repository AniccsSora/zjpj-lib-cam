class Queue:
    def __init__(self, limit):
        self.queue = []
        self.limit = limit

    def enqueue(self, item):
        self.queue.insert(0, item)  # 在列表的開頭插入新元素
        if len(self.queue) > self.limit:
            self.queue.pop()  # 如果隊列超過限制大小，移除最後一個元素

    def peek(self, position=0):
        if position < 0 or position >= len(self.queue):
            print("指定位置超出隊列範圍。")
            return -1
        if not self.is_empty():
            return self.queue[position]
        else:
            print("隊列為空。")
            return -1

    def is_empty(self):
        return len(self.queue) == 0


if __name__ == "__main__":
    q = Queue(5)
    elements = [1, 2, 3, 4, 5]  # 初始元素
    for element in elements:
        q.enqueue(element)
        print(f"插入元素: {element}, 當前隊列: {q.queue}")

    print(f"隊首元素: {q.peek()}")  # 查看隊首元素
    print(f"第3個元素: {q.peek(2)}")  # 查看第3個元素(從0開始計數)

    # 加入新元素
    q.enqueue(87)
    print(f"插入元素: 87, 當前隊列: {q.queue}")
    print(f"隊首元素: {q.peek()}")  # 再次查看隊首元素
    print(f"第4個元素: {q.peek(3)}")  # 查看第4個元素

    # 繼續加入新元素來觀察隊列變化
    q.enqueue(99)
    print(f"插入元素: 99, 當前隊列: {q.queue}")
    print(f"隊首元素: {q.peek()}")  # 再次查看隊首元素
    print(f"第5個元素: {q.peek(4)}")  # 查看隊列最後一個元素