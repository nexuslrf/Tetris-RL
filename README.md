# Tetris-RL
Course project for CSC2515 2020 Fall

Some open source environments:

* [ ] https://github.com/Kautenja/gym-tetris
* [ ] https://github.com/lusob/gym-tetris
* [ ] https://github.com/jaybutera/tetrisRL

Some existing projects:

* [ ] http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf
* [ ] https://melax.github.io/tetris/tetris.html
* [ ] https://gitlab.cs.washington.edu/xkcd/deeprl-tetris/tree/master/GA3C
* [ ] https://candlend.cn/tetrisai/ 🎯

## Ruofan's work

* [x] Build an environment for modern [Tetris](https://github.com/nexuslrf/MaTris). 
* [ ] Start my simple DQN experiment. Two issues:
    * Action space? Right now I set 34 [actions](https://github.com/nexuslrf/MaTris/blob/master/actions.py) via the combination of rotation, movement, hold, drop...
    * Observation space? Consider a matrix containing current dropped tetrominoes and current dropping one, plus next tetromnino and held tetromino

        matrix shape [3,20,10] or [20,30] or [20,10] + 2 x [4,4]