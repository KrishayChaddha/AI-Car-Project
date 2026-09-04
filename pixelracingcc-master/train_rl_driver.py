import math
import random
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend (opens a window)
import matplotlib.pyplot as plt

ACTIONS = [-1.0, 0.0, 1.0]

def make_track(length=220):
    base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.1, 0.8, -0.4, -1.2, 1.4, 0.5, -1.5, 1.7, -0.9, 0.4, 1.0, -1.3, 1.2]
    track = []
    for i in range(length):
        if i < len(base):
            track.append(base[i % len(base)])
        else:
            track.append(base[(i // 5) % len(base)] * (0.6 + 0.4 * math.sin(i / 8.0)))
    return track

class TrackEnv:
    def __init__(self):
        self.track = make_track()
        self.max_steps = 600
        self.max_speed = 1.0
        self.car_count = 8
        self.reset()

    def reset(self):
        self.offset = 0.0
        self.speed = 0.55
        self.progress = 0
        self.ticks = 0
        self.dodges = 0
        self.collisions = 0
        self.off_road = False
        self.episode_done = False
        self.cars = []
        for _ in range(self.car_count):
            self._spawn_car()
        return self._state_key()

    def _spawn_car(self):
        lane = random.choice([-0.8, -0.3, 0.0, 0.3, 0.8])
        z = random.randint(30, 120)
        speed = random.uniform(0.3, 0.8)
        self.cars.append({"lane": lane, "z": z, "speed": speed, "passed": False})

    def _move_cars(self):
        for car in self.cars:
            car["z"] -= (self.speed * 0.8 + car["speed"])
            if car["z"] < -8:
                car["z"] = random.randint(60, 170)
                car["lane"] = random.choice([-0.8, -0.3, 0.0, 0.3, 0.8])
                car["speed"] = random.uniform(0.3, 0.8)
                car["passed"] = False

    def _nearest_car_gap(self):
        best = 999.0
        best_lane = 0.0
        for car in self.cars:
            if 0.0 < car["z"] < best:
                best = car["z"]
                best_lane = car["lane"]
        return best, best_lane

    def _state_key(self):
        curve = self.track[min(self.progress, len(self.track) - 1)]
        gap, lane = self._nearest_car_gap()
        offset_bucket = int(round(max(-1.5, min(1.5, self.offset)) * 3.0))  # -4..4
        curve_bucket = int(round(max(-2.0, min(2.0, curve)) * 2.0))         # -4..4
        gap_bucket = 0 if gap > 8.0 else (1 if gap > 3.0 else (2 if gap > 1.0 else 3))
        speed_bucket = int(round(max(0.0, min(1.0, self.speed)) * 4.0))    # 0..4
        lane_bucket = int(round(lane / 0.4))
        lane_bucket = max(-2, min(2, lane_bucket))
        return (offset_bucket, curve_bucket, gap_bucket, speed_bucket, lane_bucket)

    def _count_dodge(self, car):
        """Returns True exactly once per car, the moment it finishes passing
        the player while the player was in its danger zone (a real dodge)."""
        if car["z"] < -1.0 and not car["passed"]:
            dodged = abs(self.offset - car["lane"]) < 0.6
            if dodged:
                self.dodges += 1
            car["passed"] = True
            car["z"] = random.randint(70, 170)
            car["lane"] = random.choice([-0.8, -0.3, 0.0, 0.3, 0.8])
            car["speed"] = random.uniform(0.3, 0.8)
            return dodged
        return False

    def step(self, action):
        if self.episode_done:
            self.reset()

        self.ticks += 1
        self.progress += 1
        self.progress = min(self.progress, len(self.track) - 1)

        curve = self.track[self.progress]
        self.offset += action * 0.18
        self.offset += curve * 0.05
        self.offset = max(-1.8, min(1.8, self.offset))

        self.speed = max(0.15, min(self.max_speed, self.speed + 0.03 - 0.04 * abs(action) + 0.02 * (1.0 - abs(self.offset))))
        if abs(self.offset) > 1.3:
            self.speed *= 0.94

        self._move_cars()

        reward = 0.0
        reward += 0.1  # base progress
        reward += self.speed * 0.4  # speed reward
        reward -= 0.8 * abs(self.offset)  # stay near center
        reward -= 0.3 * abs(curve)  # avoid hard curves

        done = False

        if abs(self.offset) > 1.7:
            reward -= 15.0
            self.collisions += 1
            self.off_road = True
            self.episode_done = True
            done = True

        # --- FIXED reward shaping around nearby traffic ---
        # Previously: being aligned with a car at gap in [0.85, 1.2) paid a
        # bigger reward (+3.5) than being safely in a different lane (+0.5),
        # which taught the agent to stay on a collision course. Now: staying
        # aligned while a car closes in is a continuous penalty (gives an
        # early signal to move), and the only positive reward for handling
        # traffic comes from actually completing a dodge (see below).
        nearest_gap, nearest_lane = self._nearest_car_gap()
        aligned = abs(self.offset - nearest_lane) < 0.35

        if nearest_gap < 3.0 and aligned:
            closeness = max(0.0, (3.0 - nearest_gap) / 3.0)
            reward -= 1.5 * closeness

        if nearest_gap < 0.85 and aligned:
            reward -= 12.0
            self.collisions += 1
            self.episode_done = True
            done = True

        dodge_bonus = 0.0
        for car in self.cars:
            if self._count_dodge(car):
                dodge_bonus += 4.0
        reward += dodge_bonus

        if self.progress >= len(self.track) - 1:
            reward += 20.0
            done = True
            self.episode_done = True

        if self.ticks >= self.max_steps:
            done = True
            self.episode_done = True

        next_state = self._state_key()
        return next_state, reward, done

class QLearner:
    def __init__(self, action_space=ACTIONS):
        self.action_space = action_space
        self.q = defaultdict(lambda: {a: 0.0 for a in action_space})
        self.alpha = 0.15
        self.gamma = 0.95
        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995  # slower decay -> more exploration before exploiting

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        qvals = self.q[state]
        best = max(qvals.values())
        choices = [a for a, v in qvals.items() if abs(v - best) < 1e-9]
        return random.choice(choices)

    def update(self, state, action, reward, next_state):
        q_values = self.q[state]
        next_max = max(self.q[next_state].values())
        q_values[action] += self.alpha * (reward + self.gamma * next_max - q_values[action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def run_episode(learner, env):
    state = env.reset()
    total_reward = 0.0
    for _ in range(env.max_steps):
        action = learner.choose_action(state)
        next_state, reward, done = env.step(action)
        learner.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    return env.dodges, total_reward

def moving_average(values, window=20):
    if len(values) < window:
        return []
    out = []
    running_sum = sum(values[:window])
    out.append(running_sum / window)
    for i in range(window, len(values)):
        running_sum += values[i] - values[i - window]
        out.append(running_sum / window)
    return out

def train(num_episodes=800):
    env = TrackEnv()
    learner = QLearner()
    episode_dodges = []
    episode_rewards = []
    for episode in range(1, num_episodes + 1):
        dodges, total_reward = run_episode(learner, env)
        episode_dodges.append(dodges)
        episode_rewards.append(total_reward)
        learner.decay_epsilon()
        if episode % 25 == 0:
            recent_avg = sum(episode_dodges[-25:]) / min(25, len(episode_dodges))
            print(f"Episode {episode:03d} | dodges={dodges} | avg(25)={recent_avg:.1f} | "
                  f"total_reward={total_reward:.1f} | epsilon={learner.epsilon:.3f}")
    return episode_dodges, episode_rewards

def plot_metrics(episode_dodges, episode_rewards, output_path='rl_training_metrics.png'):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_dodges, color='tab:green', linewidth=1.0, alpha=0.35, label='Dodges')
    ma_dodges = moving_average(episode_dodges, window=20)
    if ma_dodges:
        plt.plot(range(19, 19 + len(ma_dodges)), ma_dodges, color='tab:green', linewidth=2.2, label='20-ep avg')
    plt.title('Cars Dodged per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Dodges')
    plt.legend()
    plt.grid(alpha=0.25)

    plt.subplot(1, 2, 2)
    plt.plot(episode_rewards, color='tab:blue', linewidth=1.0, alpha=0.35, label='Reward')
    ma_rewards = moving_average(episode_rewards, window=20)
    if ma_rewards:
        plt.plot(range(19, 19 + len(ma_rewards)), ma_rewards, color='tab:blue', linewidth=2.2, label='20-ep avg')
    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved training graph to {output_path}")
    plt.show()  # Opens the interactive window

def main():
    episode_dodges, episode_rewards = train(800)
    plot_metrics(episode_dodges, episode_rewards)
    print('Best episode dodges:', max(episode_dodges))
    print('Average dodges (all):', sum(episode_dodges) / len(episode_dodges))
    print('Average dodges (last 50):', sum(episode_dodges[-50:]) / min(50, len(episode_dodges)))

if __name__ == '__main__':
    main()