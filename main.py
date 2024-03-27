import solutions

def main():
    min_stack = solutions.MinStack()
    min_stack.push(-2)
    min_stack.push(0)
    min_stack.push(-3)
    min_stack.getMin() # return -3
    min_stack.pop()
    min_stack.top()    # return 0
    min_stack.getMin() # return -2

if __name__ == "__main__":
    main()