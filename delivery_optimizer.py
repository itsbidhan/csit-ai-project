import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

class DeliveryOptimizer:
    def __init__(self, state_size=4, action_size=1, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = batch_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def get_state(self, current_loc, next_loc):
        return np.array([[
            current_loc['x'], current_loc['y'],
            next_loc['x'], next_loc['y']
        ]])

    def get_reward(self, current_loc, next_loc):
        distance = np.sqrt((current_loc['x'] - next_loc['x'])**2 + 
                           (current_loc['y'] - next_loc['y'])**2)
        return -distance  # Negative because we want to minimize distance

    def train(self, locations, episodes=1000):
        for e in range(episodes):
            state = self.get_state(locations[0], locations[1])  # Start from depot
            total_reward = 0
            done = False
            step = 0

            while not done:
                action = self.act(state)
                next_location = locations[action + 1]  # +1 because 0 is depot
                reward = self.get_reward(locations[step], next_location)
                next_state = self.get_state(locations[step], next_location)
                
                done = (step == len(locations) - 2)  # -2 because we don't include return to depot here
                
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                step += 1

                if done:
                    print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {self.epsilon:.2}")
                    if (e + 1) % 10 == 0:
                        self.update_target_model()
                
                self.replay()

    def optimize_route(self, locations):
        num_locations = len(locations)
        route = [0]  # Start from the depot
        unvisited = list(range(1, num_locations))

        while unvisited:
            current = route[-1]
            state = self.get_state(locations[current], locations[unvisited[0]])
            action = np.argmax(self.model.predict(state)[0])
            next_loc = unvisited[action]
            
            route.append(next_loc)
            unvisited.remove(next_loc)

        route.append(0)  # Return to depot
        stats = self.calculate_stats(route, locations)
        return route, stats

    def calculate_stats(self, route, locations):
        total_distance = 0
        for i in range(len(route) - 1):
            loc1 = locations[route[i]]
            loc2 = locations[route[i+1]]
            distance = np.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['y'] - loc2['y'])**2)
            total_distance += distance

        return {
            'total_distance': total_distance,
            'num_locations': len(locations)
        }

# Usage example:
if __name__ == "__main__":
    optimizer = DeliveryOptimizer()
    
    # Example locations
    locations = [
        {'id': 0, 'x': 0, 'y': 0, 'type': 'depot'},
        {'id': 1, 'x': 1, 'y': 5, 'type': 'delivery'},
        {'id': 2, 'x': 2, 'y': 3, 'type': 'delivery'},
        {'id': 3, 'x': 5, 'y': 1, 'type': 'delivery'},
        {'id': 4, 'x': 3, 'y': 4, 'type': 'delivery'},
    ]
    
    # Train the model
    optimizer.train(locations, episodes=1000)
    
    # Optimize route
    optimized_route, stats = optimizer.optimize_route(locations)
    
    print("Optimized Route:", optimized_route)
    print("Stats:", stats)