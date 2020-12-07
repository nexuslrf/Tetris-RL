from MaTris.gym_matris import MatrisEnv

env = MatrisEnv(no_display=False)
state = env.reset()
for i in range(1000):
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    print(f"Reward: {reward}")
    # print(state)
    # if i == 500:
    #     print("stop!")
    if done:
        env.reset()
        
print("Game over!")