import numpy as np

class SumTree:
    def __init__(self, capacity):
        """
        Initialize a Sum-Tree with a fixed capacity.
        :param capacity: Maximum number of elements in the tree (power of 2 is recommended).
        """
        self.capacity = capacity  # Maximum number of leaf nodes (transitions)
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaves
        self.data = np.zeros(capacity, dtype=object)  # Store actual transitions
        self.write_index = 0  # Points to the next leaf to overwrite
        self.size = 0  # Current size of the tree

    def add(self, priority, data):
        """
        Add a new transition with a given priority.
        :param priority: Priority of the new transition.
        :param data: The transition (state, action, reward, next_state, done).
        """
        # Determine the index of the leaf node
        leaf_index = self.write_index + self.capacity - 1

        # Update the leaf node with the new priority
        self.data[self.write_index] = data
        self.update(leaf_index, priority)

        # Update the write index and size
        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf_index, priority):
        """
        Update the priority of a leaf node and propagate the change up the tree.
        :param leaf_index: Index of the leaf node.
        :param priority: New priority value.
        """
        change = priority - self.tree[leaf_index]
        self.tree[leaf_index] = priority

        # Propagate the change up the tree
        parent_index = (leaf_index - 1) // 2
        while parent_index >= 0:
            self.tree[parent_index] += change
            if parent_index == 0:
                break
            parent_index = (parent_index - 1) // 2

    def get_leaf(self, value):
        """
        Retrieve the leaf node corresponding to the given value.
        :param value: A cumulative sum value for sampling (0 <= value <= total_priority).
        :return: (index, priority, data)
        """
        parent_index = 0

        # Traverse the tree to find the leaf
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):  # Reached leaf
                leaf_index = parent_index
                break
            else:  # Traverse left or right based on the value
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        """Return the total sum of priorities (root of the tree)."""
        return self.tree[0]
