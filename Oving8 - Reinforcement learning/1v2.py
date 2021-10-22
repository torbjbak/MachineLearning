import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')


LEARNING_RATE = 0.1
DISCOUNT = 0.95
RUNS = 50000
SHOW_EVERY = 1000
UPDATE_EVERY = 100

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = RUNS // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

def create_bins_and_q_table():
    numBins = 20
    obsSpaceSize = len(env.observation_space.high)

    bins = [
        np.linspace(-4.8, 4.8, numBins),
        np.linspace(-4, 4, numBins),
        np.linspace(-.418, .418, numBins),
        np.linspace(-4, 4, numBins)
    ]

    qTable = np.random.uniform(low=-2, high=0, size=([numBins] * obsSpaceSize + [env.action_space.n]))

    return bins, obsSpaceSize, qTable

def get_discrete_state(state, bins, obsSpaceSize):
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(stateIndex)


bins, obsSpaceSize, qTable = create_bins_and_q_table()

previousCount = []
metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}

for run in range(RUNS):
    discreteState = get_discrete_state(env.reset(), bins, obsSpaceSize)
    done = False
    count = 0

    while not done:
        if run % SHOW_EVERY == 0:
            env.render()

        count += 1
        if np.random.random() > epsilon:
            action = np.argmax(qTable[discreteState])
        else:
            action = np.random.randint(0, env.action_space.n)

        newState, reward, done, _ = env.step(action) 
        newDiscreteState = get_discrete_state(newState, bins, obsSpaceSize)

        maxFutureQ = np.max(qTable[newDiscreteState]) 
        currentQ = qTable[discreteState + (action, )]

        if done and count < 200:
            reward = -375

        newQ = (1 - LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ)
        qTable[discreteState + (action, )] = newQ
        discreteState = newDiscreteState

    previousCount.append(count)

    if END_EPSILON_DECAYING >= run >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    if run % UPDATE_EVERY == 0:
        latestRuns = previousCount[-UPDATE_EVERY:]
        averageCnt = sum(latestRuns) / len(latestRuns)
        metrics['ep'].append(run)
        metrics['avg'].append(averageCnt)
        metrics['min'].append(min(latestRuns))
        metrics['max'].append(max(latestRuns))
        print("Run:", run, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns))

env.close()

plt.plot(metrics['ep'], metrics['avg'], label="average rewards")
plt.plot(metrics['ep'], metrics['min'], label="min rewards")
plt.plot(metrics['ep'], metrics['max'], label="max rewards")
plt.legend(loc=4)
plt.show()