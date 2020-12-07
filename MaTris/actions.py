"""
A list of discrete actions
"""
import itertools

MOVEMENT_DIR = [
    "right", "left", "right-press", "left-press",
]

ROTATION_DIR = [
    "forward", "reverse",
]

DROP_METHOD = [
    "sonic drop", "hard drop"
]

item2tuple = lambda x: [(i,) for i in x]

ACTIONS = item2tuple(MOVEMENT_DIR + ROTATION_DIR + DROP_METHOD) + \
    [combo for combo in itertools.product(MOVEMENT_DIR, ROTATION_DIR)] + \
    [("no op",),] # ("hold",)]

# ACTIONS = item2tuple(MOVEMENT_DIR)
# print(ACTIONS)

