"""
TO-DO:
    Add data structures and their implementations
    1.  Static and dynamic arrays (explain how it works and that Python only
        have dynamic arrays - list())
    2.  Stack
    3.  Hashing (explain difference between normal dict and defaultdict)
    4.  Linked List (single and double linked)
    5.  Trees (Binary Search Trees, Binary Trees)
    6.  Tries
    7.  Heap (min_heap, and max_heap, explain how to create it, how it works
        and how to create max_heap from implemented in Pythob heapq, which is
        min_heap)
    8.  1-D and 2-D DP tables
    9.  Intervals
    10. Graphs
    11. Advanced Graphs
""" 


from typing import Optional, TypeAlias

class LinkedListNode():
    LinkedListNode: TypeAlias = list
    def __init__(self, val=0, next=None) -> Optional[LinkedListNode]:
        self.val = val
        self.next = next

class LinkedList():
    LinkedList: TypeAlias = list
    def __init__(self, head=None, tail=None) -> Optional[LinkedList]:
        self.head = head
        self.tail = tail

    def build(self, data):
        # for v in data:
        new_node = LinkedListNode(val=data, next=None)
        if self.head is None:
            self.head = new_node
            return

        # Linked List traversal
        current_node = self.head
        while(current_node.next):
            current_node = current_node.next

        current_node.next = new_node
            
# LL1 = LinkedListNode(val=1)
# LL2 = LinkedListNode(val=2)
# LL3 = LinkedListNode(val=3)
# LL1.next = LL2
# LL2.next = LL3

# curr = LL1
# while curr is not None:
#     print(curr.val)
#     curr = curr.next


x = LinkedList()
# curr = x.head
# while curr is not None:
#     print(curr.val)
#     curr = curr.next
# x.build(data=[1,2,3,4])
for i in [1,2,3,4,5]:
    x.build(data=i)

print(x.head.val)
y = x.head.next
print(y.val)
z = y.next.next
print(z.val)
