import numpy as np

class QLearning:
    def __init__(self,
                 model_sizes,
                 compression_rates,
                 learning_rate=0.1,
                 discount_factor=0.99,
                 exploration_rate=1.0,
                 exploration_decay=0.995,
                 min_exploration_rate=0.01):
        """
        初始化 Q-Learning 算法
        :param model_sizes: 可选的模型大小列表
        :param compression_rates: 可选的 prompt 压缩率列表
        :param learning_rate: 学习率
        :param discount_factor: 折扣因子
        :param exploration_rate: 初始探索率
        :param exploration_decay: 探索率衰减因子
        :param min_exploration_rate: 最小探索率
        """
        self.model_sizes = model_sizes
        self.compression_rates = compression_rates
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # 初始化 Q 表
        self.q_table = np.zeros((len(model_sizes), len(compression_rates), 4))

    def get_state(self, model_size, compression_rate):
        """
        将模型大小和压缩率映射为状态索引
        """
        model_index = self.model_sizes.index(model_size)
        compression_index = self.compression_rates.index(compression_rate)
        return model_index, compression_index

    def choose_action(self, state):
        """
        根据当前状态选择动作（ε-greedy 策略）
        """
        if np.random.rand() < self.exploration_rate:
            # 探索：随机选择动作
            return np.random.choice(4)
        else:
            # 利用：选择 Q 值最高的动作
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """
        更新 Q 表
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_exploration_rate(self):
        """
        衰减探索率
        """
        self.exploration_rate = max(self.exploration_rate * self.exploration_decay, self.min_exploration_rate)

    def train(self, episodes, environment):
        """
        训练 Q-Learning 算法
        :param episodes: 训练的总轮数
        :param environment: 环境函数，用于获取奖励和下一个状态
        """
        for episode in range(episodes):
            # 初始化状态
            model_size = max(self.model_sizes)
            compression_rate = max(self.compression_rates)
            state = self.get_state(model_size, compression_rate)

            done = False
            while not done:
                # 选择动作
                action = self.choose_action(state)

                # 执行动作，获取奖励和下一个状态
                next_model_size, next_compression_rate, reward, done = environment(model_size, compression_rate, action)
                next_state = self.get_state(next_model_size, next_compression_rate)

                # 更新 Q 表
                self.update_q_table(state, action, reward, next_state)

                # 更新状态
                state = next_state
                model_size = next_model_size
                compression_rate = next_compression_rate

            # 衰减探索率
            self.decay_exploration_rate()

            # 打印训练进度
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Exploration Rate: {self.exploration_rate:.4f}")

    def get_optimal_action(self, model_size, compression_rate):
        """
        获取最优动作
        """
        state = self.get_state(model_size, compression_rate)
        return np.argmax(self.q_table[state])

# 示例环境函数
def environment(model_size, compression_rate, action):
    """
    环境函数，根据当前状态和动作返回下一个状态和奖励
    :param model_size: 当前模型大小
    :param compression_rate: 当前 prompt 压缩率
    :param action: 选择的动作（0: 增大模型大小，1: 减小模型大小，2: 增大压缩率，3: 减小压缩率）
    :return: 下一个模型大小，下一个压缩率，奖励，是否结束
    """
    model_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # model sizes
    compression_rates = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
                     1900, 2000]  # Example compression rates

    # 模拟奖励函数（根据延迟和精度计算奖励）
    def calculate_reward(model_size, compression_rate):
        # 假设延迟与模型大小和压缩率成正比，精度与模型大小和压缩率成反比
        accuracy = 1 / model_size + compression_rate
        # Placeholder for inference speed (e.g., inversely proportional to model size)
        time_first = 0.04 * model_size + 0.37 * compression_rate + 76.04
        time_per = 0.12 * model_size + 23.8

        # Combine factors into a single reward
        reward = accuracy + time_first + time_per
        return reward

    # 更新状态
    if action == 0 and model_size < max(model_sizes):
        model_size = model_sizes[model_sizes.index(model_size) + 1]
    elif action == 1 and model_size > min(model_sizes):
        model_size = model_sizes[model_sizes.index(model_size) - 1]
    elif action == 2 and compression_rate < max(compression_rates):
        compression_rate = compression_rates[compression_rates.index(compression_rate) + 1]
    elif action == 3 and compression_rate > min(compression_rates):
        compression_rate = compression_rates[compression_rates.index(compression_rate) - 1]

    reward = calculate_reward(model_size, compression_rate)
    done = False  # 假设训练不会提前结束

    return model_size, compression_rate, reward, done

# 示例运行
model_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # model sizes
compression_rate = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
                 1900, 2000]  # Example compression rates
agent = QLearning(model_sizes, compression_rate)
agent.train(episodes=10, environment=environment)

# 获取最优动作
optimal_action = agent.get_optimal_action(10, 100)
print(f"Optimal action for model size 256 and compression rate 0.3: {optimal_action}")