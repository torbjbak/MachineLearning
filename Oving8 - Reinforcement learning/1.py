import numpy as np
import gym
import math

env = gym.make("CartPole-v0")

learning_rate = 0.1
discount = 0.95
episodes = 60000
total_reward = 0
prior_reward = 0
UPDATE_EVERY = 2000

Observation = [30, 30, 50, 50]
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

epsilon = 1.0

epsilon_decay_value = 0.99995

q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
q_table.shape

def get_discrete_state(state):
    discrete_state = state / np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(np.int_))

for episode in range(episodes + 1):
    reset = env.reset()
    discrete_state = get_discrete_state(reset) 
    done = False
    episode_reward = 0
    max_reward = 0

    if episode % UPDATE_EVERY == 0 and episode > 0: 
        print("Episode: " + str(episode))

    while not done: 

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
 
        #step action to get new states, reward, and the "done" status.
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward #add the reward
        
        new_discrete_state = get_discrete_state(new_state)

        if episode % UPDATE_EVERY == 0: #render
            env.render()

        if not done: #update q-table
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

    if epsilon > 0.05: #epsilon modification
        if episode_reward > prior_reward and episode > 10000:
            epsilon = math.pow(epsilon_decay_value, episode - 10000)

            if episode % UPDATE_EVERY == 0:
                print("Epsilon: " + str(epsilon))

    if episode_reward > max_reward:
        max_reward = episode_reward

    total_reward += episode_reward #episode total reward
    prior_reward = episode_reward


    #every 2000 episodes print the average time and the average reward
    if episode % UPDATE_EVERY == 0 and episode != 0:
        mean_reward = total_reward / UPDATE_EVERY
        print("Mean reward: " + str(mean_reward) + " Max reward: " + str(max_reward))
        total_reward = 0

env.close()