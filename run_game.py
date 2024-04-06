import fire
from jericho import FrotzEnv
from termcolor import cprint

from text_agent.agents import AgentInterface, HumanAgent, RandomAgent, RawHistoryAgent


def clean_initial_observation(observation: str) -> str:
    strip = """Copyright (c) 1981, 1982, 1983 Infocom, Inc. All rights reserved.
ZORK is a registered trademark of Infocom, Inc.
Revision 88 / Serial number 840726"""

    return observation.replace(strip, "").strip()


def run_with_agent(agent: AgentInterface):
    env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")

    reward = 0
    done = False
    observation, info = env.reset()
    observation = clean_initial_observation(observation)
    move_number = 0

    while not done:
        move_number += 1
        valid_actions = env.get_valid_actions(use_parallel=False)
        cprint(f"Move {move_number}:", "yellow")
        cprint(f"Reward and Score: {reward} {info['score']}", "yellow")
        cprint(f"Observation: {observation}", "green")
        cprint(f"Valid Actions: {valid_actions}", "blue")
        chosen_action = agent.choose_next_action(observation, valid_actions, reward, info["score"])
        cprint(f"Agent chose: {chosen_action}", "cyan")
        observation, reward, done, info = env.step(chosen_action)

    cprint(f"Final Observation: {observation}", "green")
    cprint(f"Final Reward and Score: {reward} {info['score']}", "magenta")
    print("Game Over! Scored", info["score"], "out of", env.get_max_score())


def run(agent_type: str):
    if agent_type == "human":
        agent = HumanAgent()
    elif agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "raw-history":
        agent = RawHistoryAgent()
    else:
        raise ValueError("Invalid agent type")

    try:
        run_with_agent(agent)
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Exiting...")


if __name__ == "__main__":
    fire.Fire(run)
