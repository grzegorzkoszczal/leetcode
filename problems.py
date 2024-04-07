import math
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
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
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
