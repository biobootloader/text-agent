import fire
from dotenv import load_dotenv
from jericho import FrotzEnv
from termcolor import cprint

from text_agent.agents import (
    AgentInterface,
    HumanAgent,
    RandomAgent,
    RawHistoryAgent,
    SummaryAgent,
    ThinkingAgent,
    MentalMapAgent,
    MentalMapAgentB,
)


def clean_initial_observation(observation: str) -> str:
    strip = """Copyright (c) 1981, 1982, 1983 Infocom, Inc. All rights reserved.
ZORK is a registered trademark of Infocom, Inc.
Revision 88 / Serial number 840726"""

    return observation.replace(strip, "").strip()


def run_with_agent(agent: AgentInterface):
    env = FrotzEnv("z-machine-games-master/jericho-game-suite/zork1.z5")

    done = False
    observation, info = env.reset()
    observation = clean_initial_observation(observation)
    move_number = 0
    reward = 0
    score = 0
    chosen_action = "start"

    while not done:
        move_number += 1
        valid_actions = env.get_valid_actions(use_parallel=False)
        cprint(f"Move {move_number}:", "yellow")
        cprint(f"Reward and Score: {reward} {info['score']}", "yellow")
        cprint(f"Observation: {observation}", "green")
        cprint(f"Valid Actions: {valid_actions}", "blue")
        agent.show_state(chosen_action, observation, reward, score, valid_actions)
        chosen_action = agent.choose_next_action()
        cprint(f"Agent chose: {chosen_action}", "cyan")
        observation, reward, done, info = env.step(chosen_action)
        score = info["score"]

    cprint(f"Final Observation: {observation}", "green")
    cprint(f"Final Reward and Score: {reward} {score}", "magenta")
    print("Game Over! Scored", score, "out of", env.get_max_score())


def run(agent_type: str):
    load_dotenv()

    if agent_type == "human":
        agent = HumanAgent()
    elif agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "raw-history":
        agent = RawHistoryAgent()
    elif agent_type == "thinking":
        agent = ThinkingAgent()
    elif agent_type == "mentalmap":
        agent = MentalMapAgent()
    elif agent_type == "agentb":
        agent = MentalMapAgentB()
    else:
        raise ValueError("Invalid agent type")

    try:
        run_with_agent(agent)
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Exiting...")


if __name__ == "__main__":
    fire.Fire(run)
