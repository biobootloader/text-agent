from jericho import FrotzEnv

# Create the environment, optionally specifying a random seed
env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")
initial_observation, info = env.reset()
done = False
while not done:
    # Take an action in the environment using the step fuction.
    # The resulting text-observation, reward, and game-over indicator is returned.
    observation, reward, done, info = env.step("open mailbox")
    # Total score and move-count are returned in the info dictionary
    print("Total Score", info["score"], "Moves", info["moves"])
print("Scored", info["score"], "out of", env.get_max_score())
