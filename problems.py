import math
import heapq
from heapq import *
from collections import deque, defaultdict
from typing import Optional, TypeAlias

ListNode: TypeAlias = list
TreeNode: TypeAlias = list


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


"""
704. Binary Search
Topics: Array, Binary Search
"""


class Solution:
    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            pivot = (left + right) // 2
            if nums[pivot] == target:
                return pivot
            elif nums[pivot] < target:
                left = pivot + 1
            else:
                right = pivot - 1
        return -1


"""
74. Search a 2D Matrix
Topics: Array, Binary Search, Matrix
"""


class Solution:
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        row_length = len(matrix[0]) - 1
        left, right = 0, row_length

        current_row = 0
        while left <= right and current_row < len(matrix):
            if current_row == 0 and target < matrix[current_row][0]:
                return False
            elif target > matrix[current_row][row_length]:
                current_row += 1
            else:
                pivot = (left + right) // 2
                if matrix[current_row][pivot] == target:
                    return True
                elif matrix[current_row][pivot] < target:
                    left = pivot + 1
                else:
                    right = pivot - 1
        return False


"""
374. Guess Number Higher or Lower
Topics: Binary Search, Interactive
"""


# The guess API is already defined for you.
# @param num, your guess
# @return -1 if num is higher than the picked number
#          1 if num is lower than the picked number
#          otherwise return 0
# def guess(num: int) -> int:
def guess(pivot):
    secret_target = 5
    if pivot > secret_target:
        return -1
    elif pivot < secret_target:
        return 1
    else:
        return 0


class Solution:
    # def binary_search(self) -> int:
    #     pass

    def guessNumber(self, n: int) -> int:
        left, right = 0, n

        while left <= right:
            pivot = (left + right) // 2
            feedback = guess(pivot)
            if (
                feedback < 0
            ):  # guess(pivot) return -1 if num (pivot) > target (picked number)
                right = pivot - 1
            elif (
                feedback > 0
            ):  # guess(pivot) return 1 if num (pivot) < target (picked number)
                left = pivot + 1
            else:
                return pivot
        return -1


# guess_number = Solution()
# print(guess_number.guessNumber(100))

"""
278. First Bad Version
Topics: Binary Search, Interactive
"""


# The isBadVersion API is already defined for you.
def isBadVersion(version: int) -> bool:
    pass


class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 0, n

        if n == 1:
            return 1
        while left <= right:
            pivot = (left + right) // 2
            feedback = isBadVersion(pivot)
            # print(f"ver. {pivot}. Is Bad version? {feedback}")
            if feedback is True:
                right = pivot
            elif feedback is False:
                left = pivot
            # print("what happend to right?", right)
            if (
                isBadVersion(left) is False
                and isBadVersion(right) is True
                and left == (right - 1)
            ):
                return right


"""
205. Isomorphic Strings
Topics: Hash Table, String
"""


class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        dict_s, dict_t = dict(), dict()
        for i, j in zip(s, t):
            if i not in dict_s.keys():
                dict_s[i] = j
            if j not in dict_t.keys():
                dict_t[j] = i
            if (i in dict_s.keys() or j in dict_t.keys()) and (
                dict_s[i] != j or dict_t[j] != i
            ):
                return False
        #     print(i, j)
        # print(dict_s)
        # print(dict_t)
        return True


"""
875. Koko Eating Bananas
Topics: Array, Binary Search
"""


class Solution:
    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        piles.sort()
        max_speed = piles[-1]
        min_speed = 1
        ans = max_speed

        while min_speed <= max_speed:
            pivot = (min_speed + max_speed) // 2
            time = 0
            for pile in piles:
                time += math.ceil(pile / pivot)
            if time <= h:
                ans = min(ans, pivot)
                max_speed = pivot - 1
            else:
                min_speed = pivot + 1
        return ans


"""
28. Find the Index of the First Occurrence in a String
Topics: Two Pointers, String, String Matching
"""


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        ans = -1
        flag = len(needle)
        if needle not in haystack:
            return ans
        for idx, ch in enumerate(haystack):
            if haystack[idx] == needle[0]:
                ans = idx
                for i in range(len(needle)):
                    if haystack[idx + i] == needle[i]:
                        print(haystack[i])
                        flag -= 1
                    else:
                        ans = -1
                        break
                if flag == 0:
                    return ans
                else:
                    flag = len(needle)
        return ans


"""
79. Word Search
Topics: Array, String, Backtracking, Matrix
"""


class Solution:
    def exist(self, board: list[list[str]], word: str) -> bool:
        ROWS, COLS, memo = len(board), len(board[0]), set()

        def dfs(r, c, idx):
            if idx == len(word):
                return True
            if (
                r not in range(ROWS)
                or c not in range(COLS)
                or (r, c) in memo
                or word[idx] != board[r][c]
            ):
                return False

            memo.add((r, c))
            if (
                dfs(r + 1, c, idx + 1)
                or dfs(r - 1, c, idx + 1)
                or dfs(r, c - 1, idx + 1)
                or dfs(r, c + 1, idx + 1)
            ):
                return True
            memo.remove((r, c))

        for r in range(ROWS):
            for c in range(COLS):
                if dfs(r, c, idx=0):
                    return True
        return False


"""
700. Search in a Binary Search Tree
Topics: Tree, Binary Search Tree, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # 1. Solution - recursive
        if not root:
            return
        if val < root.val:
            return self.searchBST(root.left, val)
        elif val > root.val:
            return self.searchBST(root.right, val)
        else:
            return root

        # 2. Solution - iterative
        while root:
            if val < root.val:
                root = root.left
            elif val > root.val:
                root = root.right
            else:
                return root
        return None


"""
701. Insert into a Binary Search Tree
Topics: Tree, Binary Search Tree, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        old_root = root
        while root:
            if val < root.val and root.left:
                root = root.left
            elif val > root.val and root.right:
                root = root.right
            else:
                if val < root.val:
                    root.left = TreeNode(val)
                    return old_root
                else:
                    root.right = TreeNode(val)
                    return old_root
        return TreeNode(val)


"""
450. Delete Node in a BST
Topics: Tree, Binary Search Tree, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def find_min(self, root):
        while root and root.left:
            root = root.left
        return root

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None

        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        elif key > root.val:
            root.right = self.deleteNode(root.right, key)
        else:
            # target node has 0 or 1 child
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            else:
                leaf = self.find_min(root.right)
                root.val = leaf.val
                root.right = self.deleteNode(root.right, leaf.val)
        return root


"""
94. Binary Tree Inorder Traversal
Topics: Stack, Tree, Depth-First Search, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> list[int]:
        if not root:
            return []
        left = self.inorderTraversal(root.left)
        right = self.inorderTraversal(root.right)
        return left + [root.val] + right  # inorder traversal
        # return [root.val] + left + right # preorder traversal
        # return left + right + [root.val] # postorder traversal


"""
1614. Maximum Nesting Depth of the Parentheses
Topics: String, Stack
"""


class Solution:
    def maxDepth(self, s: str) -> int:
        temp, ans = 0, 0
        for ch in s:
            if ch == "(":
                temp += 1
            elif ch == ")":
                temp -= 1
            ans = max(ans, temp)
        return ans


"""
230. Kth Smallest Element in a BST
Topics: Tree, Depth-First Search, Binary Search Tree, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def helper(root):
            if not root:
                return []
            return helper(root.left) + [root.val] + helper(root.right)

        ans = helper(root)
        return ans[k - 1]


"""
1544. Make The String Great
Topics: String, Stack
"""


class Solution:
    def makeGood(self, s: str) -> str:
        """
        a, z, A, Z = "a", "z", "A", "Z"
        print(f"a: {ord(a)}, z: {ord(z)}, A: {ord(A)}, Z: {ord(Z)}")
        a: 97, z: 122, A: 65, Z: 90
        """
        stack = list()
        ans = ""
        for ch in s:
            # lowercase on stack, uppercase in string
            if (
                stack
                and ord(stack[-1]) in range(97, 123)
                and ord(ch) in range(65, 91)
                and stack[-1] == ch.lower()
            ):
                stack.pop()
                ans = ans[:-1]
            # uppercase on stack, lowercase in string
            elif (
                stack
                and ord(stack[-1]) in range(65, 91)
                and ord(ch) in range(97, 123)
                and stack[-1] == ch.upper()
            ):
                stack.pop()
                ans = ans[:-1]
            else:
                stack.append(ch)
                ans += ch
        return ans

        # Similar, but cleaned-up solution
        # stack = []
        # for char in s:
        #     if stack and abs(ord(stack[-1]) - ord(char)) == 32:
        #         stack.pop()  # Remove the previous character
        #     else:
        #         stack.append(char)
        # return ''.join(stack)


"""
1249. Minimum Remove to Make Valid Parentheses
Topics: String, Stack
"""


class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        count = 0
        ans = list(s)

        for i, ch in enumerate(ans):
            if ch == "(":
                count += 1
            elif ch == ")":
                if count == 0:
                    ans[i] = "#"
                else:
                    count -= 1
        ans.reverse()

        for i, ch in enumerate(ans):
            if count > 0 and ch == "(":
                ans[i] = "#"
                count -= 1
        ans.reverse()

        ans = "".join(x for x in ans if x != "#")
        return ans


"""
678. Valid Parenthesis String
Topics: String, Dynamic Programming, Stack, Greedy
"""


class Solution:
    def checkValidString(self, s: str) -> bool:
        left_min, left_max = 0, 0
        for i in s:
            if i == "(":
                left_min += 1
                left_max += 1
            elif i == ")":
                left_min -= 1
                left_max -= 1
            else:
                left_min -= 1
                left_max += 1
            if left_max < 0:
                return False
            if left_min < 0:
                left_min = 0
        return True if left_min == 0 else False


"""
9. Palindrome Number
Topics: Math
"""


class Solution:
    def isPalindrome(self, x: int) -> bool:
        s = str(x)
        left, right = 0, len(s) - 1
        while left <= right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return False
        return True


"""
35. Search Insert Position
Topics: Array, Binary Search
"""


class Solution:
    def searchInsert(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            pivot = (left + right) // 2
            if nums[pivot] == target:
                return pivot
            elif nums[pivot] < target:
                left = pivot + 1
            elif nums[pivot] > target:
                right = pivot - 1
        return left


"""
105. Construct Binary Tree from Preorder and Inorder Traversal
Topics: Array, Hash Table, Divide and Conquer, Tree, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: list[int], inorder: list[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1 :], inorder[mid + 1 :])
        return root


"""
102. Binary Tree Level Order Traversal
Topics: Tree, Breadth-First Search, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> list[list[int]]:
        if not root:
            return []

        ans, temp, q = list(list()), list(), deque()
        q.append(root)

        while q:
            for _ in range(len(q)):
                curr = q.popleft()
                temp.append(curr.val)
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)
            ans.append(temp)
            temp = list()

        return ans


"""
199. Binary Tree Right Side View
Topics: Tree, Depth-First Search, Breadth-First Search, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> list[int]:
        if not root:
            return []

        ans, temp, q = list(), list(), deque()
        q.append(root)

        while q:
            for _ in range(len(q)):
                curr = q.popleft()
                temp.append(curr.val)
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)
            ans.append(temp[-1])
            temp = list()
        return ans


"""
1700. Number of Students Unable to Eat Lunch
Topics: Array, Stack, Queue, Simulation
"""


class Solution:
    def countStudents(self, students: list[int], sandwiches: list[int]) -> int:
        # 0 - circular sandwich, 1 - square sandwich
        students_q, sandwiches_q = deque(students.copy()), deque(sandwiches.copy())
        breaker = len(students_q)
        while students_q and breaker > 0:
            if students_q[0] == sandwiches_q[0]:
                students_q.popleft()
                sandwiches_q.popleft()
                breaker = len(students_q)
            else:
                temp = students_q.popleft()
                students_q.append(temp)
                breaker -= 1
        return len(students_q)


"""
112. Path Sum
Topics: Tree, Depth-First Search, Breadth-First Search, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def helper(root, current_sum):
            if not root:
                return False

            current_sum += root.val
            if not root.left and not root.right:
                return current_sum == targetSum

            return helper(root.left, current_sum) or helper(root.right, current_sum)

        return helper(root, current_sum=0)


"""
78. Subsets
Topics: Array, Backtracking, Bit Manipulation
"""


class Solution:
    def subsets(self, nums: list[int]) -> list[list[int]]:
        ans = list()
        subset = list()

        def backtracking(i):
            if i >= len(nums):
                ans.append(subset.copy())
                return

            # decision to include nums[i]
            subset.append(nums[i])
            backtracking(i + 1)

            # decision to NOT include nums[i]
            subset.pop()
            backtracking(i + 1)

        backtracking(0)
        return ans


"""
39. Combination Sum
Topics: Array, Backtracking
"""


class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        ans = list()

        def dfs(i, curr, total):
            if total == target:
                ans.append(curr.copy())
                return

            if total > target or i >= len(candidates):
                return

            curr.append(candidates[i])
            dfs(i, curr, total + candidates[i])
            curr.pop()
            dfs(i + 1, curr, total)

        dfs(0, [], 0)
        return ans


"""
1025. Divisor Game
Topics: Math, Dynamic Programming, Brainteaser, Game Theory
"""


class Solution:
    def divisorGame(self, n: int) -> bool:
        # 1. One-liner solution
        # return n % 2 == 0

        # 2. Dynamic Programming solution
        # memory
        dp = [False] * (n + 1)
        # establish base case for n == 1
        dp[0] = True
        dp[1] = False
        # visit every situation and update memory if make for winning move
        for s in range(2, n + 1):
            for f in range(1, s):
                if dp[s - f] == False and s % f == 0:
                    dp[s] = True
        # return cumulative solution from memory
        print(dp)
        return dp[n]


"""
257. Binary Tree Paths
Topics: String, Backtracking, Tree, Depth-First Search, Binary Tree
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> list[str]:
        ans = list()

        def dfs(node, path):
            if node:
                path = path + str(node.val) + "->"
                # print(f"current node {node.val}")
            if not node:
                return [""]
            if not node.left and not node.right:
                ans.append(path[:-2])
                return

            dfs(node.left, path)
            dfs(node.right, path)

        dfs(root, "")
        return ans


"""
401. Binary Watch
Topics: Backtracking, Bit Manipulation
"""


class Solution:
    def readBinaryWatch(self, turnedOn: int) -> list[str]:
        ans = list()
        led = [8, 4, 2, 1, 32, 16, 8, 4, 2, 1]

        # how many digits are still on
        def dfs(h, m, idx, n):
            if h > 11 or m > 59:
                return
            if n == 0:
                ans.append("{:d}:{:02d}".format(h, m))
                return
            for i in range(
                idx, len(led)
            ):  # <- right here, we can just iterate through the options we have
                if i <= 3:
                    dfs(h + led[i], m, i + 1, n - 1)
                elif i < len(led):
                    dfs(h, m + led[i], i + 1, n - 1)

        dfs(0, 0, 0, turnedOn)
        return ans


"""
2708. Maximum Strength of a Group
Topics: Array, Backtracking, Greedy, Sorting
"""


class Solution:
    def maxStrength(self, nums: list[int]) -> int:
        curr_max = curr_min = nums[0]
        for i in nums[1:]:
            temp_max = max(i, curr_min * i, curr_max * i, curr_max)
            curr_min = min(i, curr_min * i, curr_max * i, curr_min)
            curr_max = temp_max
        return curr_max


"""
797. All Paths From Source to Target
Topics: Backtracking, Depth-First Search, Breadth-First Search, Graph
"""


class Solution:
    def allPathsSourceTarget(self, graph: list[list[int]]) -> list[list[int]]:
        ans = list()
        end_node = len(graph) - 1
        q = deque([[0]])

        while q:
            temp = q.popleft()
            if temp[-1] == end_node:
                ans.append(temp)
            for neighbor in graph[temp[-1]]:
                q.append(temp + [neighbor])
        return ans


"""
2073. Time Needed to Buy Tickets
Topics: Array, Queue, Simulation
"""


class Solution:
    def timeRequiredToBuy(self, tickets: list[int], k: int) -> int:
        q = deque(tickets.copy())
        time = 0
        while q:
            # print(f"time: {time}, q: {q}, k: {k}")
            q[0] -= 1
            if q[0] == 0:
                if k == 0:
                    return time + 1
                q.popleft()
                k -= 1
            else:
                if k == 0:
                    k = len(q) - 1
                else:
                    k -= 1
                temp = q.popleft()
                q.append(temp)
            time += 1
        return time


"""
2000. Reverse Prefix of Word
Topics: Two Pointers, String
"""


class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        for i, c in enumerate(word):
            if c == ch:
                return word[i::-1] + word[i + 1 : :]
        return word


"""
950. Reveal Cards In Increasing Order
Topics: Array, Queue, Sorting, Simulation
"""


class Solution:
    def deckRevealedIncreasing(self, deck: list[int]) -> list[int]:
        deck.sort()

        size = len(deck)
        ans = [0] * size
        indices = deque(range(size))

        for card in deck:
            idx = indices.popleft()
            ans[idx] = card
            if indices:
                indices.append(indices.popleft())

        return ans

"""
703. Kth Largest Element in a Stream
Topics: Tree, Design, Binary Search Tree, Heap (Priority Queue), Binary Tree, Data Stream
"""
class KthLargest:

    def __init__(self, k: int, nums: list[int]):
        self.nums = nums[:k]
        self.k = k
        heapify(self.nums)
        for i in range(k, len(nums)):
            if nums[i] > self.nums[0]:
                heappushpop(self.nums, nums[i])

    def add(self, val: int) -> int:
        if len(self.nums) < self.k:
            heappush(self.nums, val)
        elif val > self.nums[0]:
            heapreplace(self.nums, val)

        return self.nums[0]


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)

"""
496. Next Greater Element I
Topics: Array, Hash Table, Stack, Monotonic Stack
"""
class Solution:
    def nextGreaterElement(self, nums1: list[int], nums2: list[int]) -> list[int]:
        ans = [-1] * len(nums1)

        d = dict()
        left, right = 0, 1
        while left <= right and right < len(nums2):
            if nums2[left] < nums2[right]:
                d[nums2[left]] = nums2[right]
                left += 1
                right = left + 1
            elif right == len(nums2)-1:
                d[nums2[left]] = -1
                left += 1
                right = left + 1
            else:
                right += 1

        for i, v in enumerate(nums1):
            if v in d.keys():
                ans[i] = d[v]

        return ans
    
"""
1475. Final Prices With a Special Discount in a Shop
Topics: Array, Stack, Monotonic Stack
"""
class Solution:
    def finalPrices(self, prices: list[int]) -> list[int]:
        mono_stack = list()
        for i in range(len(prices)):
            while mono_stack and (prices[mono_stack[-1]] >= prices[i]):
                prices[mono_stack.pop()] -= prices[i]
            mono_stack.append(i)
        return prices
    
"""
402. Remove K Digits
Topics: String, Stack, Greedy, Monotonic Stack
"""
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        mono = list()

        for i in num:
            while mono and mono[-1] > i and k > 0:
                mono.pop()
                k -= 1
            mono.append(i)

        if k > 0:
            mono = mono[:-k]

        ans = "".join(mono).lstrip("0")
        return ans if ans else "0"
    
"""
1046. Last Stone Weight
Topics: Array, Heap (Priority Queue)
"""
class Solution:
    def lastStoneWeight(self, stones: list[int]) -> int:
        stones = [-1 * x for x in stones]
        heapq.heapify(stones)
        while len(stones) > 1:
            heapq.heappush(stones, -abs(heapq.heappop(stones) - heapq.heappop(stones)))
        return -1 * stones[0] if stones else 0

"""
973. K Closest Points to Origin
Topics: Array, Math, Divide and Conquer, Geometry, Sorting, Heap (Priority Queue), Quickselect
"""
class Solution:
    def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
        for i in range(len(points)):
            dist = pow(points[i][0], 2) + pow(points[i][1], 2)
            points[i].append(dist)
        points.sort(key=lambda x: x[2])
        ans = [[points[i][0], points[i][1]] for i in range(k)]

        return ans
    
"""
42. Trapping Rain Water
Topics: Array, Two Pointers, Dynamic Programming, Stack, Monotonic Stack
"""
class Solution:
    def trap(self, height: list[int]) -> int:
        left, right = 0, len(height)-1
        left_max, right_max = height[0], height[right]
        ans = 0
        while left < right:
            if left_max <= right_max:
                ans += left_max - height[left]
                left += 1
                left_max = max(left_max, height[left])
            else:
                ans += right_max - height[right]
                right -= 1
                right_max = max(right_max, height[right])
        return ans

"""
217. Contains Duplicate
Topics: Array, Hash Table, Sorting
"""
class Solution:
    def containsDuplicate(self, nums: list[int]) -> bool:
        s = set()
        for i in nums:
            if i not in s:
                s.add(i)
            else:
                return True
        return False
    
"""
1. Two Sum
Topics: Array, Hash Table
"""
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        d = dict()
        for i, v in enumerate(nums):
            if (target - v) in d:
                return [i, d[target - v]]
            d[v] = i

"""
85. Maximal Rectangle
Topics: Array, Dynamic Programming, Stack, Matrix, Monotonic Stack
"""
class Solution:
    def maximalRectangle(self, matrix: list[list[str]]) -> int:
        ROWS, COLS = len(matrix), len(matrix[0])
        dp = dict()
        ans = 0

        for r in range(ROWS):
            for c in range(COLS):
                if matrix[r][c] == "0":
                    dp[(r, c)] = (0, 0)
                else:
                    x = dp[r, c-1][0]+1 if c > 0 else 1
                    y = dp[r-1, c][1]+1 if r > 0 else 1
                    dp[(r, c)] = (x, y)
                    ans = max(x, y, ans)
                    min_width = x
                    for i in range(r-1, r-y, -1):
                        min_width = min(min_width, dp[(i, c)][0])
                        ans = max(ans, min_width * (r-i+1))
        return ans

"""
146. LRU Cache
Topics: Hash Table, Linked List, Design, Doubly-Linked List
"""
class LinkedListNode:
    def __init__(self, key=0, val=0, prev=None, next=None):
        self.key = key
        self.val = val
        self.prev = prev
        self.next = next

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = dict()
        self.head = LinkedListNode(key=None, val="head", prev=None, next=None)
        self.tail = LinkedListNode(key=None, val="tail", prev=None, next=None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def insert(self, node):
        temp = self.tail.prev
        self.tail.prev = node
        node.next = self.tail
        node.prev = temp
        temp.next = node

    def delete(self, node):
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev

    def put(self, key: int, value: int) -> None:
        if key in self.cache.keys():
            self.delete(self.cache[key])
        self.cache[key] = LinkedListNode(key=key, val=value)
        self.insert(self.cache[key])

        if len(self.cache) > self.capacity:
            lru = self.head.next
            self.delete(lru)
            del self.cache[lru.key]

    def get(self, key: int) -> int:
        if key in self.cache.keys():
            self.delete(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        else:
            return -1 

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
        
        
"""
404. Sum of Left Leaves
Topics: Tree, Depth-First Search, Breadth-First Search, Binary Tree
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)

            if root.left and not root.left.left and not root.left.right:
                left += root.left.val
            return left + right

        return dfs(root)
    
"""
129. Sum Root to Leaf Numbers
Topics: Tree, Depth-First Search, Binary Tree
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        cache, stack, ans = list(), str(), int()
        def dfs(root, stack, cache):
            if root:
                stack += str(root.val)
            if root and not root.left and not root.right:
                cache.append(stack)
                stack = stack[:-1]
            if not root:
                return None

            left = dfs(root.left, stack, cache)
            right = dfs(root.right, stack, cache)
            return left, right

        dfs(root, stack, cache)
        for i in cache:
            ans += int(i)

        return ans

"""
623. Add One Row to Tree
Topics: Tree, Depth-First Search, Breadth-First Search, Binary Tree
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
        if depth == 1:
            return TreeNode(val=val, left=root, right=None)

        queue = deque([root])
        while queue:
            depth -= 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if depth == 1:
                    node.left = TreeNode(val, left=node.left, right=None)
                    node.right = TreeNode(val, left=None, right=node.right)
            if depth == 1:
                break
        return root
    
"""
705. Design HashSet
Topics: Array, Hash Table, Linked List, Design, Hash Function
"""
class MyHashSet:
    def __init__(self):
        self.storage = [0 for _ in range(1000001)]

    def add(self, key: int) -> None:
        if self.storage[key] == 0:
            self.storage[key] = 1

    def remove(self, key: int) -> None:
        if self.storage[key] == 1:
            self.storage[key] = 0

    def contains(self, key: int) -> bool:
        return True if self.storage[key] == 1 else False

# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
    
"""
706. Design HashMap
Topics: Array, Hash Table, Linked List, Design, Hash Function
"""
class Node:
    def __init__(self, key=None, value=None, next=None):
        self.key = key
        self.value = value
        self.next = next

class MyHashMap:
    def __init__(self):
        self.size = 1000
        self.hash_table = [None] * self.size

    def put(self, key: int, value: int) -> None:
        index = key % self.size
        if self.hash_table[index] == None:
            self.hash_table[index] = Node(key, value)
        else:
            curr_node = self.hash_table[index]
            while True:
                if curr_node.key == key:
                    curr_node.value = value
                    return
                if curr_node.next == None: break
                curr_node = curr_node.next
            curr_node.next = Node(key, value)

    def get(self, key: int) -> int:
        index = key % self.size
        curr_node = self.hash_table[index]
        while curr_node:
            if curr_node.key == key:
                return curr_node.value
            else:
                curr_node = curr_node.next
        return -1

    def remove(self, key: int) -> None:
        index = key % self.size
        curr_node = prev_node = self.hash_table[index]
        if not curr_node: return
        if curr_node.key == key:
            self.hash_table[index] = curr_node.next
        else:
            curr_node = curr_node.next
            while curr_node:
                if curr_node.key == key:
                    prev_node.next = curr_node.next
                    break
                else:
                    prev_node, curr_node = prev_node.next, curr_node.next


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
                    
"""
988. Smallest String Starting From Leaf
Topics: String, Tree, Depth-First Search, Binary Tree
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        temp, ans = str(), list()
        def dfs(root, temp, ans):
            if not root:
                return None
            if root:
                current_char = chr(ord("a") + root.val)
                temp += current_char
            if root and not root.left and not root.right:
                path = temp[::-1]
                if not ans:
                    ans.append(path)
                else:
                    old, new = ans[0], path
                    if new < old:
                        ans[0] = new

            left = dfs(root.left, temp, ans)
            right = dfs(root.right, temp, ans)
            return left, right

        dfs(root, temp, ans)
        return ans[0]

"""
1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree
Topics: Tree, Depth-First Search, Breadth-First Search, Binary Tree
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        ans = list()
        def dfs(root, target, ans):
            if not root:
                return
            if root.val == target.val:
                ans.append(root)
                return

            left = dfs(root.left, target, ans)
            right = dfs(root.right, target, ans)
            return left, right

        dfs(cloned, target, ans)
        return ans[0]
    
"""
563. Binary Tree Tilt
Topics: Tree, Depth-First Search, Binary Tree
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findTilt(self, root: Optional[TreeNode]) -> int:
        self.ans = 0
        def dfs(root):
            if not root:
                return 0

            left = dfs(root.left)
            right = dfs(root.right)
            self.ans += abs(left - right)
            return root.val + left + right

        dfs(root)
        return self.ans
    
"""
463. Island Perimeter
Topics: Array, Depth-First Search, Breadth-First Search, Matrix
"""
class Solution:
    def islandPerimeter(self, grid: list[list[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])
        visited, ans = set(), 0

        def dfs(r, c):
            if (not r in range(ROWS) or
                not c in range(COLS) or
                grid[r][c] == 0):
                return 1
            if (r, c) in visited:
                return 0

            visited.add((r, c))
            ans = dfs(r +1, c)
            ans += dfs(r -1, c)
            ans += dfs(r, c +1)
            ans += dfs(r, c -1) 
            return ans

        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c]:
                    return dfs(r, c)