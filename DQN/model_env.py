class ModelEnvironment:
    def __init__(self, model_sizes, prompt_length):
        self.model_sizes = model_sizes
        self.prompt_length = prompt_length
        self.state_space = [(size, prompt) for size in model_sizes for prompt in prompt_length]
        self.action_space = [0, 1, 2, 3] # 0: extend size; 1: reduce size; 2: extend radio; 3: reduce radio
        self.current_state = (len(self.model_sizes) - 1, len(self.prompt_length) - 1)  # Initial state

    def reset(self):
        self.current_state = (len(self.model_sizes) - 1, len(self.prompt_length) - 1)
        return self.current_state

    def step(self, action):
        model_idx, comp_idx = self.current_state

        if action == 0 and model_idx < len(self.model_sizes) - 1:
            model_idx = model_idx + 1
        elif action == 1 and model_idx > 0:
            model_idx = model_idx - 1
        elif action == 2 and comp_idx < len(self.prompt_length) - 1:
            comp_idx = comp_idx + 1
        elif action == 3 and comp_idx > 0:
            comp_idx = comp_idx - 1

        self.current_state = (model_idx, comp_idx)
        reward = self.calculate_reward(model_idx, comp_idx)

        return self.current_state, reward

    def calculate_reward(self, model_idx, comp_idx):
        model_size = self.model_sizes[model_idx]
        prompt_length = self.prompt_length[comp_idx]

        # Placeholder for accuracy (e.g., based on model size and compression rate)
        accuracy = 1 / model_size + prompt_length

        # Placeholder for inference speed (e.g., inversely proportional to model size)
        time_first = 0.04 * model_size + 0.37 * prompt_length + 76.04
        time_per = 0.12 * model_size + 23.8

        # Combine factors into a single reward
        reward = accuracy + time_first + time_per
        return reward