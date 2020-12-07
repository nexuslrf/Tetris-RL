from MaTris.gym_matris import MatrisEnv

env = MatrisEnv(no_display=False)
for i in range(1000):
    state, reward, done, info = env.step(env.action_space.sample())
    print(f"Reward: {reward}")
    if not done:
        env.render()
    else:
        env.reset()
        
print("Game over!")