import numpy as np
import random

class QLearningAgent:
    def __init__(self, model_sizes, compression_rates, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        初始化Q-learning代理
        :param model_sizes: 模型大小的可能值列表
        :param compression_rates: Prompt压缩率的可能值列表
        :param learning_rate: 学习率
        :param discount_factor: 折扣因子
        :param epsilon: 探索率
        """
        self.model_sizes = model_sizes
        self.compression_rates = compression_rates
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # 初始化Q表，状态为延迟和精度的组合，动作是模型大小和压缩率的组合
        self.q_table = np.zeros((len(model_sizes), len(compression_rates)))

    def choose_action(self, state):
        """
        根据当前状态选择动作
        :param state: 当前状态，格式为 (延迟, 精度)
        :return: 选择的模型大小和压缩率
        """
        if random.uniform(0, 1) < self.epsilon:
            # 探索：随机选择动作
            model_size_idx = random.randint(0, len(self.model_sizes) - 1)
            compression_rate_idx = random.randint(0, len(self.compression_rates) - 1)
        else:
            # 利用：选择Q值最高的动作
            model_size_idx, compression_rate_idx = np.unravel_index(
                np.argmax(self.q_table), self.q_table.shape
            )

        return model_size_idx, compression_rate_idx

    def update_q_table(self, state, action, reward, next_state):
        """
        更新Q表
        :param state: 当前状态
        :param action: 采取的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        """
        model_size_idx, compression_rate_idx = action
        best_next_action = np.argmax(self.q_table)
        best_next_q_value = self.q_table.ravel()[best_next_action]

        # 更新Q值
        self.q_table[model_size_idx, compression_rate_idx] += self.learning_rate * (
            reward + self.discount_factor * best_next_q_value - self.q_table[model_size_idx, compression_rate_idx]
        )

    def get_action(self, state):
        """
        根据当前状态选择最优动作
        :param state: 当前状态
        :return: 选择的模型大小和压缩率
        """
        model_size_idx, compression_rate_idx = np.unravel_index(
            np.argmax(self.q_table), self.q_table.shape
        )
        return self.model_sizes[model_size_idx], self.compression_rates[compression_rate_idx]

def reward_function(delay, accuracy, target_delay, target_accuracy):
    """
    计算奖励函数
    :param delay: 实际延迟
    :param accuracy: 实际精度
    :param target_delay: 目标延迟
    :param target_accuracy: 目标精度
    :return: 奖励值
    """
    delay_reward = -abs(delay - target_delay) / target_delay
    accuracy_reward = -abs(accuracy - target_accuracy) / target_accuracy
    return delay_reward + accuracy_reward

# 示例：模型大小和压缩率的可能值
model_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # model sizes
compression_rates = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
                     1900, 2000]  # Example compression rates

# 初始化Q-learning代理
agent = QLearningAgent(model_sizes, compression_rates)

# 模拟训练过程
num_episodes = 1000
for episode in range(num_episodes):
    # 用户输入的目标延迟和精度
    target_delay = random.uniform(0.1, 1.0)  # 示例：随机生成目标延迟
    target_accuracy = random.uniform(0.8, 1.0)  # 示例：随机生成目标精度

    # 当前状态
    state = (target_delay, target_accuracy)

    # 选择动作
    model_size_idx, compression_rate_idx = agent.choose_action(state)
    model_size = model_sizes[model_size_idx]
    compression_rate = compression_rates[compression_rate_idx]

    # 模拟实际延迟和精度（这里可以根据实际模型进行计算）
    actual_delay = random.uniform(0.1, 1.0)  # 示例：随机生成实际延迟
    actual_accuracy = random.uniform(0.8, 1.0)  # 示例：随机生成实际精度

    # 计算奖励
    reward = reward_function(actual_delay, actual_accuracy, target_delay, target_accuracy)

    # 更新Q表
    agent.update_q_table(state, (model_size_idx, compression_rate_idx), reward, state)

    # 打印训练过程
    print(f"Episode {episode + 1}: Target Delay={target_delay:.2f}, Target Accuracy={target_accuracy:.2f}, "
          f"Chosen Model Size={model_size}, Chosen Compression Rate={compression_rate}, "
          f"Reward={reward:.2f}")

# 测试
target_delay = 0.5
target_accuracy = 0.9
state = (target_delay, target_accuracy)
optimal_model_size, optimal_compression_rate = agent.get_action(state)
print(f"Optimal Model Size: {optimal_model_size}, Optimal Compression Rate: {optimal_compression_rate}")