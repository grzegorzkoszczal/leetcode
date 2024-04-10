"""
This is a cheatsheet to store the most useful boilerplate code snippets
that can be reused for future problems


Topics list:
    Data structures:
        1. Arrays (lists)
        2. Strings
        3. Dynamic programming
        4. Linked lists
        5. 

    Algorithms:
        1. Two pointers
        2. Kadane's Algorithm
        3. Depth First Search
        4. Breadth First Search
        5. Dijkstra's Algorithm
        6. Binary search

    Sorting algorithms:
        1. Bubble sort
        2. Insertion sort
        3. Quicksort
        4. Merge sort       - comparison-based, Time: O(n log n), Space: O(n)
        5. Timsort

        
TO-DO LIST:
    1. Linked list class
    2. Binary Tree class
"""

from collections import defaultdict, deque
from typing import Generator, TypeAlias, Union, Optional


class Git:
    def __init__(self) -> None:
        pass

    def info(self) -> None:
        """
        git init
            ->  create a new repository in the current directory. The name
                of repository is based of the name of current directory.
        git clone <name-of-the-repo>
            ->  bring a repository that is hosted on GitHub into a folder on
                local machine
        git status
            ->  check modified and untracked files
        git pull
            ->  if you have ".git" folder, which contains all information
                required for version control, you can download all missing
                files from remote repository
        git add (if you want to add all changes and untracked files,
                use "git add .")
            ->  track your files and changes in Git
        git commit -m "title" -m "description"
            ->  save your files in Git
            ->  if you have problem with "Author identity unknown", type this:
                ->  git config --global user.email "you@example.com"
                ->  git config --global user.name "Your Name"
        git push
            -> Upload Git commits to a remote repo (GitHub)
        """
        pass


class Python:
    def __init__(self) -> None:
        self.x = 0  # some test variable without any real purpose, ignore

    proper_knowledge: TypeAlias = bool

    def proper_documentation(self, read: bool = False) -> proper_knowledge:
        """
        Explain the function purpose/intention NOT implementation:
        This function servers no real purpose other than to learn the correct
        principles of documenting the functions in Python.

        Parameters:
        read (bool): set to False, because You did not read it yet

        Returns:
        bool = True, because You read it and learned something new
        """
        read = True
        return read

    def sandbox(self):
        print(self.x)
        return 0

    def decorator(func):
        """
        Additional notes:
            Decorators defined inside class don't need "self"
        """

        def wrapper(*args, **kwargs):
            print("I am a decorator!")
            print(f'Calling function: "{func.__name__}"')
            ans = func(*args, **kwargs)
            return ans

        return wrapper

    def class_decorator(cls):
        print(f"Calling class: {cls.__name__}")
        return cls

    def generator(self, count_up_to: int) -> Generator[int, None, None]:
        internal_counter = 1
        while internal_counter <= count_up_to:
            yield internal_counter
            internal_counter += 1

    @staticmethod
    def add(var1: Union[int, float], var2: Union[int, float]) -> Union[int, float]:
        """
        Purpose: Learn static method.

        Parameters:
        var1 (can be "int" or "float"): first parameter to calculate.
        var2 (can be "int" or "float"): second parameter to calculate.

        Return:
        ans (can be "int" or "float", based on input data): return the sum.

        Additional notes:
            Static methods don't need "self"
            You dont have to instantiate the class, You can simply type
            "Python.add(3, 5)" in main function and it will work.
        """
        ans = var1 + var2
        print(ans)
        return ans


class TestCases:
    def __init__(self):
        self.nums = [1, 3, 4, 7, 9]
        self.matrix = [[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 0], [0, 0, 1, 1]]
        self.words = "airspace intelligence"

    def dynamic_programming(self):
        pass

    def string(self) -> str:
        cases = list()

        case_1 = {
            "input": {"list_of_strings": ["eat", "tea", "tan", "ate", "nat", "bat"]},
            "output": [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]],
        }
        cases.append(case_1)

        case_2 = {"input": {"list_of_strings": [""]}, "output": [[""]]}
        cases.append(case_2)

        case_3 = {"input": {"list_of_strings": [""]}, "output": [[""]]}
        cases.append(case_3)

        return case_1
        # return cases

    # @classmethod
    def tree(self) -> list[int]:
        root = [1, None, 2, 3]
        # root = {
        #     "input": {
        #         "root": [1, None, 2, 3]
        #     },
        #     "output": [1,3,2]
        # }
        return root


class String:
    def __init__(self):
        pass

    def create_list_of_lowercase_letters(self) -> list:
        alphabet_list = [0 for _ in range(ord("z") - ord("a"))]
        # print(alphabet_list)
        return alphabet_list

    def create_dict_of_lowercase_letters(self) -> dict:
        alphabet_dict = {chr(i + ord("a")): 0 for i in range(ord("z") - ord("a") + 1)}
        # print(alphabet_dict)
        return alphabet_dict

    def count_letters_in_list(self, input_string: str) -> list:
        updated_list = self.create_list_of_lowercase_letters()

        for char in input_string:
            if char.isalpha():
                updated_list[ord(char) - ord("a")] += 1
        # print(updated_list)
        return updated_list

    def count_letters_in_dict(self, input_string: str) -> dict:
        updated_dictionary = self.create_dict_of_lowercase_letters()

        for char in input_string:
            if char.isalpha():
                updated_dictionary[char] = updated_dictionary.get(char, 0) + 1
        # print(updated_dictionary)
        return updated_dictionary


class DynamicProgramming:
    def __init__(self) -> None:
        pass

    def create_1d_list(self, nums: list) -> list:
        dp_array = [0 for _ in range(len(nums))]
        # print(dp_array)
        return dp_array

    def create_2d_list(self, matrix: list) -> list:
        dp_array = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        # print(dp_array)
        return dp_array


class TreeNode(TestCases):
    def __init__(self, val=0, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right

    TreeNode: TypeAlias = list

    def build(self, nodes: list, index: int = 0) -> Optional[TreeNode]:
        if index < len(nodes):
            if nodes[index] is None:
                return None
            root = TreeNode(nodes[index])
            root.left = self.build(nodes, 2 * index + 1)
            root.right = self.build(nodes, 2 * index + 2)
            return root
        return None


class BinarySearchTree:
    def __init__(self, root=None):
        self.root = root

    def add(self, root, value):
        if not self.root:
            self.root = TreeNode(value)
        else:
            if value < root.value:
                if root.left is None:
                    root.left = TreeNode(value)
                else:
                    self.add(root.left, value)
            else:
                if root.right is None:
                    root.right = TreeNode(value)
                else:
                    self.add(root.right, value)


class Tree:
    def __init__(self) -> None:
        pass

    def inorder_traversal(self, root: Optional[TreeNode]) -> list[int]:
        if not root:
            return []

        print(self.inorder_traversal(root.left))
        print(root.val, end=" ")
        print(self.inorder_traversal(root.right))

        # left = self.inorder_traversal(root.left)
        # right = self.inorder_traversal(root.right)

        # return left + [root.val] + right

    def bfs(self, root: Optional[TreeNode]) -> list[list[int]]:
        queue = deque()  # queue is used for storing the nodes on current level
        queue.append(root)  # appending the first node to the queue
        visited = set()  # set is used for storing all visited nodes

        while queue:  # iterate until queue is not empty
            node = queue.popleft()  # "node" becomes "root" on each level
            visited.add(node)  # add current node to visited set
            [queue.append(x) for x in root[node]]  # adding children to the queue
        print(visited)

    def dfs(self, root: Optional[TreeNode]) -> list[list[int]]:
        stack = list()  # list acts like a stack for the purpose of dfs
        stack.append(root)  # appending the first node to the stack
        visited = set()  # set is used for storing all visited nodes

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                [stack.append(x) for x in root[node]]
        print(visited)


class TrieNode:
    def __init__(self):
        self.children = dict()
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            if not c in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.end_of_word = True

    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            if not c in curr.children:
                return False
            curr = curr.children[c]
        return curr.end_of_word

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for c in prefix:
            if not c in curr.children:
                return False
            curr = curr.children[c]
        return True


@Python.class_decorator
class Solution:

    @Python.decorator
    def group_anagrams(self, list_of_strings: list[str]) -> list[list[str]]:
        ans = defaultdict(list)

        for word in list_of_strings:
            count = String.create_list_of_lowercase_letters(self)

            for char in word:
                count[ord(char) - ord("a")] += 1

            ans[tuple(count)].append(word)

        # print([*ans.values()])
        # print(list(ans.values()))
        return [*ans.values()]


import time


def fast_decorator(func):
    def wrapper(*args, **kwargs):
        print("I am a wrapper")
        start = time.time()
        time.sleep(1)
        ans = func(*args, **kwargs)
        end = time.time() - start
        print("Total execution time: {0:0.3f}, {1:0.5f}".format(end, start))
        return ans

    return wrapper


@fast_decorator
def addition(x):
    x = x + 1
    return x


def main(test=TestCases(), ans=Solution()):

    # t = Tree()
    # tn = TreeNode()

    # root = tn.build(test.tree())
    # # root = [1, None, 2, 3]
    # bt = t.inorder_traversal(root)
    # bt = t.bfs(root)
    # print(bt)

    # case = test.string()
    # y = ans.group_anagrams(**case["input"]) == case["output"]
    # print(y)
    # Python.add(3, 5)
    # c = Python()
    # print(c.proper_documentation())

    # count = z.generator(5)
    # for i in count:
    #     print(i)

    # count = z.generator(3)
    # for i in count:
    #     print(i)
    y = addition(5)
    print(y)

    # z = lambda x, y: x+y if x > 5 and y > 0 else -1
    # print(z(6, 1))


if __name__ == "__main__":
    main()
