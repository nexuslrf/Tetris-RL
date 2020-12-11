from MaTris.gym_matris_v2 import MatrisEnv
from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool as Pool
import copy

def foo(x):
    return x.matrix.sum()

mp_pool = Pool(4)
env = MatrisEnv(no_display=False, mp_pool=mp_pool)
for i in range(1000):
    state, reward, done, info = env.step(env.action_space.sample())

    diff_sum = 0

    # ret = env.mp_peek_step()
    # for i in range(env.action_space.n):
    #     s, r, d, f = env.peak_step_srdi(i)
    #     diff_sum += (ret[i][0]!=s).sum()

    print(f"Reward: {reward} | Diff_sum: {diff_sum}")
    if not done:
        env.render()
    else:
        env.reset()