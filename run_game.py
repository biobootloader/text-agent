from jericho import FrotzEnv
from termcolor import cprint


def run():
    # Create the environment, optionally specifying a random seed
    env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")
    # env = FrotzEnv("z-machine-games-master/jericho-game-suite/sherlock.z5")
    initial_observation, info = env.reset()
    print(initial_observation)  # Show the initial observation to the user
    done = False

    while not done:
        # Get and display valid actions
        valid_actions = env.get_valid_actions(use_parallel=False)
        cprint(f"Valid Actions: {valid_actions}", "green")

        # Take an action from the user input
        user_action = input("Enter your action: ")

        # Take the action in the environment using the step function.
        # The resulting text-observation, reward, and game-over indicator is returned.
        observation, reward, done, info = env.step(user_action)

        # Show the results to the user
        cprint(f"Observation: {observation}", "green")
        cprint(f"Reward: {reward}", "yellow")
        cprint(f"Done: {done}", "blue")
        cprint(f"Info: {info}", "magenta")

        # Total score and move-count are returned in the info dictionary
        print("Total Score", info["score"], "Moves", info["moves"])

    print("Game Over! Scored", info["score"], "out of", env.get_max_score())


if __name__ == "__main__":
    run()
