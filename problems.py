from collections import deque, defaultdict
from typing import Optional, TypeAlias

ListNode: TypeAlias = list


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

"""
206. Reverse Linked List
Topics: Linked List, Recursion
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp

        return prev


"""
21. Merge Two Sorted Lists
Topics: Linked List, Recursion
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        # list1 and list2 ARE heads!
        curr = head = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                curr.next = list1
                curr = list1
                list1 = list1.next
            else:
                curr.next = list2
                curr = list2
                list2 = list2.next
        if list1 or list2:
            if list1:
                curr.next = list1
            else:
                curr.next = list2
        return head.next


"""
707. Design Linked List
Topics: Linked List, Design
"""


# Definition for double-linked list.
# class ListNode:
#     def __init__(self, val=0, prev=None, next=None):
#         self.val = val
#         self.prev = prev
#         self.next = next
class Node:
    def __init__(self, val=0):
        self.val = val
        self.prev = None
        self.next = None


class MyLinkedList:
    def __init__(self):
        self.head = Node(0)
        self.tail = Node(0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, index: int) -> int:
        curr = self.head.next
        while curr and index > 0:
            curr = curr.next
            index -= 1
        if curr and curr != self.tail and index == 0:
            return curr.val
        else:
            return -1

    def addAtHead(self, val: int) -> None:
        curr, prev, next = Node(val), self.head, self.head.next
        prev.next = curr
        next.prev = curr
        curr.prev = prev
        curr.next = next

    def addAtTail(self, val: int) -> None:
        curr, prev, next = Node(val), self.tail.prev, self.tail
        prev.next = curr
        next.prev = curr
        curr.prev = prev
        curr.next = next

    def addAtIndex(self, index: int, val: int) -> None:
        curr = self.head.next
        while curr and index > 0:
            curr = curr.next
            index -= 1
        if curr and index == 0:
            insert, prev, next = Node(val), curr.prev, curr
            prev.next = insert
            next.prev = insert
            insert.prev = prev
            insert.next = next

    def deleteAtIndex(self, index: int) -> None:
        curr = self.head.next
        while curr and index > 0:
            curr = curr.next
            index -= 1
        if curr and curr != self.tail and index == 0:
            prev, next = curr.prev, curr.next
            prev.next = next
            next.prev = prev


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)


"""
1472. Design Browser History
Topics: Array, Linked List, Stack, Design, Doubly-Linked List, Data Stream
"""


class Node:
    def __init__(self, val="", prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next


class BrowserHistory:
    def __init__(self, homepage: str):
        self.position = Node(val=homepage)

    def visit(self, url: str) -> None:
        """
        1. Visits url from the current page.
        2. It clears up all the forward history.
        """
        self.position.next = Node(val=url, prev=self.position)
        self.position = self.position.next

    def back(self, steps: int) -> str:
        """
        1. Move steps back in history.
        2. If you can only return x steps in the history and steps > x,
           you will return only x steps.
        3. Return the current url after moving back in history at most steps
        """
        while self.position.prev and steps > 0:
            self.position = self.position.prev
            steps -= 1
        return self.position.val

    def forward(self, steps: int) -> str:
        """
        1. Move steps forward in history.
        2. If you can only forward x steps in the history and steps > x,
           you will forward only x steps.
        3. Return the current url after forwarding in history at most steps
        """
        while self.position.next and steps > 0:
            self.position = self.position.next
            steps -= 1
        return self.position.val


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)

# Explanation:
# browserHistory = BrowserHistory(homepage="leetcode.com")
# browserHistory.visit("google.com")       # You are in "leetcode.com". Visit "google.com"
# browserHistory.visit("facebook.com")     # You are in "google.com". Visit "facebook.com"
# browserHistory.visit("youtube.com")      # You are in "facebook.com". Visit "youtube.com"
# browserHistory.back(1)                   # You are in "youtube.com", move back to "facebook.com" return "facebook.com"
# browserHistory.back(1)                   # You are in "facebook.com", move back to "google.com" return "google.com"
# browserHistory.forward(1)                # You are in "google.com", move forward to "facebook.com" return "facebook.com"
# browserHistory.visit("linkedin.com")     # You are in "facebook.com". Visit "linkedin.com"
# browserHistory.forward(2)                # You are in "linkedin.com", you cannot move forward any steps.
# browserHistory.back(2)                   # You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
# browserHistory.back(7)                   # You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"

"""
1700. Number of Students Unable to Eat Lunch
Topics: Array, Stack, Queue, Simulation
"""


class Solution:
    def countStudents(self, students: list[int], sandwiches: list[int]) -> int:
        students_q = deque(students.copy())
        sandwiches_q = deque(sandwiches.copy())
        counter = len(students_q)
        while students_q and sandwiches_q and counter > 0:
            # print(students_q, sandwiches_q)
            # print(counter)
            if students_q[0] == sandwiches_q[0]:
                students_q.popleft()
                sandwiches_q.popleft()
                counter = len(students_q)
                continue
            else:
                temp = students_q.popleft()
                students_q.append(temp)
                counter -= 1
        return len(students_q)


# x = Solution()
# y = x.countStudents(students=[1,1,0,0], sandwiches=[0,1,0,1])
# # y = x.countStudents(students=[1,1,1,0,0,1], sandwiches=[1,0,0,0,1,1])
# print(y)

"""
225. Implement Stack using Queues
Topics: Stack, Design, Queue
"""


class MyStack:
    def __init__(self):
        self.q = list()

    def push(self, x: int) -> None:
        return self.q.append(x)

    def pop(self) -> int:
        return self.q.pop()

    def top(self) -> int:
        return self.q[-1]

    def empty(self) -> bool:
        return False if bool(self.q) else True


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()


"""
70. Climbing Stairs
Topics: Math, Dynamic Programming, Memoization
"""


class Solution:
    def climbStairs(self, n: int) -> int:
        first, second, ans = 1, 2, 1
        if n == 2:
            return 2
        for _ in range(2, n):
            ans = first + second
            first = second
            second = ans
        return ans


# 1, 2, 3, 5, 8, 13

"""
509. Fibonacci Number
Topics: Math, Dynamic Programming, Recursion, Memoization
"""


class Solution:
    def __init__(self):
        self.memo = dict()

    def fib(self, n: int) -> int:
        if n < 2:
            return n
        if n in self.memo:
            return self.memo[n]
        ans = self.fib(n - 1) + self.fib(n - 2)
        self.memo[n] = ans
        return ans


"""
912. Sort an Array
Topics: Array, Divide and Conquer, Sorting, Heap (Priority Queue)
        Merge Sort, Bucket Sort, Radix Sort, Counting Sort
"""


class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        # Merge in-place
        def merge(nums, left, pivot, right):
            # Copy the sorted left & right halfs to temp arrays
            L = nums[left : pivot + 1]
            R = nums[pivot + 1 : right + 1]

            i = 0  # index for L
            j = 0  # index for R
            k = left  # index for nums

            # Merge the two sorted halfs into the original array
            while i < len(L) and j < len(R):
                if L[i] <= R[j]:
                    nums[k] = L[i]
                    i += 1
                else:
                    nums[k] = R[j]
                    j += 1
                k += 1

            # One of the halfs will have elements remaining
            while i < len(L):
                nums[k] = L[i]
                i += 1
                k += 1
            while j < len(R):
                nums[k] = R[j]
                j += 1
                k += 1

        def merge_sort(nums, left, right):
            if (right - left + 1) <= 1:
                return nums

            pivot = (left + right) // 2
            merge_sort(nums, left, pivot)
            merge_sort(nums, pivot + 1, right)

            merge(nums, left, pivot, right)

        merge_sort(nums, left=0, right=len(nums) - 1)
        return nums


"""
23. Merge k Sorted Lists
Topics: Linked List, Divide and Conquer, Heap (Priority Queue), Merge Sort
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: list[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists or len(lists) == 0:
            return None
        while len(lists) > 1:
            merged_lists = list()

            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None
                merged_lists.append(self.merge_lists(l1, l2))
            lists = merged_lists
        return lists[0]

    def merge_lists(self, l1, l2):
        dummy = ListNode()
        tail = dummy

        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next

        if l1:
            tail.next = l1
        if l2:
            tail.next = l2
        return dummy.next


"""
58. Length of Last Word
Topics: String
"""


class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        ans = 0
        for char in reversed(s):
            if char.isalnum():
                ans += 1
                continue
            elif char == " " and ans == 0:
                continue
            elif char == " " and ans != 0:
                break
        return ans


"""
215. Kth Largest Element in an Array
Topics: Array, Divide and Conquer, Sorting, Heap (Priority Queue), Quickselect
"""


class Solution:
    def findKthLargest(self, nums: list[int], k: int) -> int:
        # 1. Solution - using TimSort from standard library:
        # nums.sort(reverse=True)

        # for i in range(k):
        #     ans = nums[i]
        # return ans
        # 2. Solution - using min_heap of size k
        import heapq

        heap = nums[:k]
        heapq.heapify(heap)

        for i in nums[k:]:
            if i > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, i)
        return heap[0]


"""
75. Sort Colors
Topics: Array, Two Pointers, Sorting (Bucket Sort)
"""


class Solution:
    def sortColors(self, nums: list[int]) -> None:
        # Algorithm used: Bucket Sort
        """
        Do not return anything, modify nums in-place instead.
        """
        counter = [0, 0, 0]
        for i in nums:
            counter[i] += 1

        print(counter)

        itr = 0
        for i in range(len(counter)):
            for _ in range(counter[i]):
                nums[itr] = i
                itr += 1
