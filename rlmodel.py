import numpy as np

states = 3
actions = 4

Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

def get_reward(state, action):
    if state == 0:
        if action in [0, 1]:
            return 10
        else:
            return -5

    elif state == 1:
        if action in [1, 3]:
            return 10
        else:
            return 5

    elif state == 2:
        if action == 2:
            return 10
        elif action == 3:
            return 5
        else:
            return -10

def train_rl():
    global Q
    for _ in range(2000):
        state = np.random.randint(0, states)

        if np.random.rand() < epsilon:
            action = np.random.randint(0, actions)
        else:
            action = np.argmax(Q[state])

        reward = get_reward(state, action)
        next_state = state

        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

def recommend_treatment(state):
    action = np.argmax(Q[state])

    treatments = {
        0: "Diet Plan",
        1: "Exercise",
        2: "Medication",
        3: "Lifestyle Changes"
    }

    return treatments[action]