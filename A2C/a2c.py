from tqdm import tqdm
from datetime import datetime
import time
from A2C.model import Agent
from shared.tensorboard_wrapper import TensorboardWrapper


def a2c(env, hyper_parameters):
    train_seconds = hyper_parameters['train_seconds']
    actor_learning_rate = hyper_parameters['actor_learning_rate']
    critic_learning_rate = hyper_parameters['critic_learning_rate']
    gamma = hyper_parameters['gamma']
    actor_layers = hyper_parameters['actor_layers']
    critic_layers = hyper_parameters['critic_layers']

    agent = Agent(actor_layers, critic_layers, actor_learning_rate, critic_learning_rate, gamma, env.device)

    tq = tqdm()

    now = datetime.now()
    now_str = now.strftime("%y%m%d-%H%M")
    tensorboard_writer = TensorboardWrapper(now_str)

    score = 0
    epoch = 0

    prev_state = env.reset()
    now = time.time()
    start_time = now
    while now - start_time < train_seconds:
        tq.update(1)

        action = agent.decide(prev_state)
        next_state, reward, done, info = env.step(action)

        actor_loss, critic_loss = agent.train(prev_state, reward, next_state, done)

        tensorboard_writer.save_scalar('actor_loss', actor_loss, env.total_steps)
        tensorboard_writer.save_scalar('critic_loss', critic_loss, env.total_steps)

        score += reward

        if done:
            prev_state = env.reset()
            tensorboard_writer.save_scalar('score', score, epoch)

            epoch += 1
            score = 0
        else:
            prev_state = next_state


        now = time.time()

    return
