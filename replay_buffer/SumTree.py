import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree is ssaved as an array format, first elements are internal nodes and then leafs
        self.data = np.zeros(capacity, dtype=object) # For actual transitions
        self.write_index = 0  # Points to the next leaf available to store
        self.size = 0

    def add(self, priority, data):
        # New leaf index
        leaf_index = self.write_index + self.capacity - 1

        # Update the leaf with priority
        self.data[self.write_index] = data
        self.update(leaf_index, priority)

        # Update the write index and size
        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf_index, priority):
        change = priority - self.tree[leaf_index]
        self.tree[leaf_index] = priority

        # Propagate new priority up to the tree root
        parent_index = (leaf_index - 1) // 2
        while parent_index >= 0:
            self.tree[parent_index] += change
            if parent_index == 0:
                break
            parent_index = (parent_index - 1) // 2

    def get_leaf(self, value):
        parent_index = 0

        # search the leaf
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):  # found leaf
                leaf_index = parent_index
                break
            else:  # traverse left or right based on the value
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]
