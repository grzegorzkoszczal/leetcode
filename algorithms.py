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
            while left >= 0 and nums[left + 1] < nums[left]:
                nums[left], nums[left + 1] = nums[left + 1], nums[left]
                left -= 1
        else:
            while left >= 0 and nums[left + 1] > nums[left]:
                nums[left], nums[left + 1] = nums[left + 1], nums[left]
                left -= 1

    return nums


# x = insertion_sort(nums, reversed=False)
# print(x)

"""
Merge Sort
Traits:
> Stable sorting algorithm,
> Best Time Complexity: O(nlogn)
> Worst Time Complexity: O(nlogn)
"""


class MergeSort:
    def merge(self, nums, left, pivot, right):
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

    def merge_sort(self, nums, left, right):
        if (right - left + 1) <= 1:
            return nums

        pivot = (left + right) // 2
        self.merge_sort(nums, left, pivot)
        self.merge_sort(nums, pivot + 1, right)

        self.merge(nums, left, pivot, right)
        return nums


ms = MergeSort()
ans = ms.merge_sort(nums, left=0, right=len(nums) - 1)
print(ans)

"""
Quick Sort
Traits:
> Unstable sorting algorithm,
> Best Time Complexity: O(nlogn)
> Worst Time Complexity: O(n^2)
"""


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        left = [x for x in arr[1:] if x < pivot]
        right = [x for x in arr[1:] if x >= pivot]
        return quicksort(left) + [pivot] + quicksort(right)


# Example usage
arr = [1, 7, 4, 1, 10, 9, -2]
sorted_arr = quicksort(arr)
print("Sorted Array in Ascending Order:")
print(sorted_arr)


"""
Bucket Sort
Traits:
> Unstable sorting algorithm,
> Best Time Complexity: O(n)
> Worst Time Complexity: O(n)
"""


def bucket_sort(bucket_arr):
    # Assuming arr only contains 0, 1 or 2
    counts = [0, 0, 0]

    # Count the quantity of each val in arr
    for n in bucket_arr:
        counts[n] += 1

    # Fill each bucket in the original array
    i = 0
    for n in range(len(counts)):
        for _ in range(counts[n]):
            bucket_arr[i] = n
            i += 1
    return bucket_arr


bucket_arr = [0, 1, 2, 2, 2, 1, 1, 0, 0, 1, 2]
ans = bucket_sort(bucket_arr)
print(ans)
