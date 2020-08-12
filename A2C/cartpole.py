import torch
from shared.gym_env import Environment
from tqdm import tqdm
from datetime import datetime
import time
from A2C.model import Agent
from shared.tensorboard_wrapper import TensorboardWrapper


def main():
    # Hyper parameters
    train_seconds = 60 * 60
    actor_learning_rate = 0.00005
    critic_learning_rate = 0.0003
    gamma = 0.99

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = Environment('CartPole-v1', device)
    prev_state = env.reset()

    agent = Agent(env.get_shape_observations(), env.get_n_actions(), actor_learning_rate, critic_learning_rate, gamma, device)

    tq = tqdm()

    now = datetime.now()
    now_str = now.strftime("%y%m%d-%H%M")
    tensorboard_writer = TensorboardWrapper(now_str)

    score = 0
    epoch = 0

    now = time.time()
    start_time = now
    while now - start_time < train_seconds:
        tq.update(1)

        action = agent.decide(prev_state)
        next_state, reward, done, info = env.step(action)

        actor_loss, critic_loss = agent.train(prev_state, reward, next_state, done)

        tensorboard_writer.save_scalar('actor_loss', actor_loss, env.total_steps)
        tensorboard_writer.save_scalar('critic_loss', critic_loss, env.total_steps)

        if done:
            prev_state = env.reset()
            tensorboard_writer.save_scalar('score', score, epoch)

            epoch += 1
            score = 0
        else:
            prev_state = next_state

        score += 1

        now = time.time()

    return


if __name__ == "__main__":
    main()