class Stack(object):
    def __init__(self):
        # 创建一个空的栈
        self.item = []

    def push(self,item):
        # 添加新元素到栈顶
        self.item.append(item)

    def pop(self):
        # 弹出栈顶元素
        return self.item.pop()

    def peek(self):
        # 返回栈顶元素
        return self.item[len(self.item)-1]

    def isEmpty(self):
        # 检验是否为空
        return self.item == []

    def size(self):
        # 返回栈的个数
        return len(self.item)