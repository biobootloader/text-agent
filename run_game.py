from jericho import FrotzEnv

# Create the environment, optionally specifying a random seed
env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")
initial_observation, info = env.reset()
print(initial_observation)  # Show the initial observation to the user
done = False
while not done:
    # Get and display valid actions
    valid_actions = env.get_valid_actions()
    print(f"Valid Actions: {valid_actions}")
    
    # Take an action from the user input
    user_action = input("Enter your action: ")
    
    # Take the action in the environment using the step function.
    # The resulting text-observation, reward, and game-over indicator is returned.
    observation, reward, done, info = env.step(user_action)
    
    # Show the results to the user
    print("Observation:", observation)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    
    # Total score and move-count are returned in the info dictionary
    print("Total Score", info["score"], "Moves", info["moves"])

print("Game Over! Scored", info["score"], "out of", env.get_max_score())
