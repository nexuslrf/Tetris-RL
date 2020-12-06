from MaTris.gym_matris import MatrisEnv

env = MatrisEnv(no_display=True)
for i in range(1000):
    reward, done, state, info = env.step(env.action_space.sample())
    print(f"Reward: {reward}")
    if not done:
        env.render()
    else:
        env.reset()
        
print("Game over!")