from __future__ import print_function
from collections import namedtuple

X, O = 'X', None
Tetromino = namedtuple("Tetrimino", "color shape")

tetrominoes = {
    "I": Tetromino(color="blue",
                      shape=((O,O,O,O),
                             (X,X,X,X),
                             (O,O,O,O),
                             (O,O,O,O))),
    "O": Tetromino(color="yellow",
                        shape=((X,X),
                               (X,X))),
    "T": Tetromino(color="pink",
                     shape=((O,X,O),
                            (X,X,X),
                            (O,O,O))),
    "S": Tetromino(color="green",
                             shape=((O,X,X),
                                    (X,X,O),
                                    (O,O,O))),
    "Z": Tetromino(color="red",
                            shape=((X,X,O),
                                   (O,X,X),
                                   (O,O,O))),
    "J": Tetromino(color="cyan",
                          shape=((X,O,O),
                                 (X,X,X),
                                 (O,O,O))),
    "L": Tetromino(color="orange",
                           shape=((O,O,X),
                                  (X,X,X),
                                  (O,O,O)))
}
list_of_tetrominoes = list(tetrominoes.keys())
list_of_start_tetrominoes = ["I", "T", "J", "L"]

color_map = {
    "blue": 1,
    "yellow": 2,
    "pink": 3,
    "green": 4,
    "red": 5,
    "cyan": 6,
    "orange": 7,
    "grey": 8,
}
colors = ["none","blue","yellow","pink","green","red","cyan","orange","grey"]

def rotate(shape, times=1):
    """ Rotate a shape to the right """
    return shape if times == 0 else rotate(tuple(zip(*shape[::-1])), times-1)


def shape_str(shape):
    """ Return a string of a shape in human readable form """
    return '\n'.join(''.join(map({'X': 'X', None: 'O'}.get, line))
                     for line in shape)

def shape(shape):
    """ Print a shape in human readable form """
    print(shape_str(shape))
def test():
    tetromino_shapes = [t.shape for t in list_of_tetrominoes]
    map(rotate,    tetromino_shapes)
    map(shape,     tetromino_shapes)
    map(shape_str, tetromino_shapes)

    assert shape_str(tetrominoes["left_snake"].shape) == "XXO\nOXX\nOOO"

    assert rotate(tetrominoes["square"].shape) == tetrominoes["square"].shape

    assert rotate(tetrominoes["right_snake"].shape, 4) == tetrominoes["right_snake"].shape

    assert rotate(tetrominoes["hat"].shape)    == ((O,X,O),
                                                   (O,X,X),
                                                   (O,X,O))

    assert rotate(tetrominoes["hat"].shape, 2) == ((O,O,O),
                                                   (X,X,X),
                                                   (O,X,O))
    print("All tests passed in {}, things seems to be working alright".format(__file__))

if __name__ == '__main__':
    test()
