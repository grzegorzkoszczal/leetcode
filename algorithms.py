"""
TO-DO:
    Add algorithms and their practical implementations
    1.  Kadane's Algorith (and where it is used)
    2.  Greedy Algorithm
    3.  BFS
    4.  DFS
    5.  Two-pointers approach
    6.  Sliding window
    7.  Binary search
    8.  Sorting algorithms listed in "boilerplates" module
"""

"""
Insertion Sort
Traits:
> Stable sorting algorithm,
> Best Time Complexity: O(n)
> Worst Time Complexity: O(n^2)
"""
nums = [2, 3, 4, 1, 6]
def insertion_sort(nums: list, reversed: bool = False) -> list: 
    for right in range(1, len(nums)):
        left = right - 1
        if reversed == False:
            while (left >= 0 and nums[left+1] < nums[left]):
                nums[left], nums[left+1] = nums[left+1], nums[left]
                left -= 1
        else:
            while (left >= 0 and nums[left+1] > nums[left]):
                nums[left], nums[left+1] = nums[left+1], nums[left]
                left -= 1

    return nums


x = insertion_sort(nums, reversed=False)
print(x)