import random

import algorithms
import data_structures

# import problems
# import test_cases
# import boilerplates


def main():
    temp = ["MinHeap"]
    min_heap = data_structures.MinHeap()
    for _ in range(20):
        val = random.randint(1, 50)
        temp.append(val)
        min_heap.add(val)
    print(f"Initial list: {temp}")
    min_heap.show()


if __name__ == "__main__":
    main()
