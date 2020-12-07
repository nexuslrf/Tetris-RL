This is modified Tetris pygame with part of modern tetris features like hold, T-Spin.

![](t_spin-demo.gif)

We also provide some gym-like interfaces for doing some Reinforcement learning experiments

```python
env = MatrisEnv()
for i in range(1000):
    reward, done, state, info = env.step(env.action_space.sample())
    print(f"Reward: {reward}")
    if not done:
        env.render()
    else:
        env.reset()  
print("Game over!") 
```

*The original readme.md are shown below.*

----

MaTris
======

A clone of Tetris made using Pygame. Licensed under the GNU GPLv3. Works with both Python 2 and 3.

Run `python matris.py` (or `python3 matris.py`) to start the game.

Requirements
============

The game requires [pygame](https://www.pygame.org). On Ubuntu it can be installed with these commands: `sudo apt install python-pip && sudo pip install pygame` (for Python 2) `sudo apt install python3-pip && sudo pip3 install pygame` (for Python 3).

Demo
====
![Demo](demo.png)

Coveted by academia
========================
In 2013, my game [was used](http://eprints.ucm.es/22631/1/REMIRTA.pdf) by someone in Madrid to test "remote execution of multimedia interactive real-time applications". The next year, [a study in Denmark](https://www.academia.edu/6262472/Improving_game_experience_using_dynamic_difficulty_adjustment_based_on_physiological_signals) called "Improving game experience using dynamic diﬃculty adjustment" asked participants to "self-rate their valence and arousal [sic]" playing MaTris! Who would've thunk it? In 2016, people in Stanford [were using the game](http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf) to try out deep reinforcement learning, although apparently the result was not as "respectable" as it could've been. Not a problem in Korea, apparently, where students [are expected](http://nlp.chonbuk.ac.kr/AML/AML_assignment_2.pdf) to accomplish it! That stuff is way above my head, but perhaps my life will be spared during the singularity?
