"""
155. Min Stack
Topics: Stack, Design
"""

class MinStack:
    def __init__(self):
        self.main_stack = list()
        self.support_stack = list()

    def push(self, val: int) -> None:
        self.main_stack.append(val)
        if self.support_stack:
            val = min(self.support_stack[-1], val)
        self.support_stack.append(val)

    def pop(self) -> None:
        self.main_stack.pop()
        self.support_stack.pop()

    def top(self) -> int:
        print("Stack top value:\t{}".format(self.main_stack[-1]))
        return self.main_stack[-1]

    def getMin(self) -> int:
        print("Stack minimum value:\t{}".format(self.support_stack[-1]))
        return self.support_stack[-1]        

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()