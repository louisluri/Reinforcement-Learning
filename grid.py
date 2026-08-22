# Monte Carlo (MC): An agent must complete an entire episode (from start to finish) 
# before it can update its values. 
# It looks at the total accumulated reward (Return) 
# from that episode to decide which actions were good.

# Monte Carlo Update:
# Q(St​,At​)←Q(St​,At​)+α[Gt​−Q(St​,At​)]
# (Where Gt​ is the actual total reward from time t to the end. Gt​ = Rt + γRt+1 + γ^2Rt+2 + γ^3Rt+3 + ...)

# Temporal Difference (TD): An agent updates its values after every single step. 
# It doesn't wait for the finish line; it learns "online" as it interacts with the environment.
# Rewards earned propagate back to the previous state-action pairs immediately, allowing for faster learning and adaptability.

# Temporal Difference (TD) Update:
# Q(St​,At​)←Q(St​,At​)+α[(Target)−Q(St​,At​)]
# (Where the Target is an estimate consisting of the immediate reward 
# plus the discounted value of the next state.)

# SARSA Target: Rt+1 + γ Q(St+1​,At+1​)
# Expected SARSA Target: Rt+1 + γ Σa π(a|St+1​) Q(St+1​,a)
# Q-learning Target: Rt+1 + γ maxa Q(St+1​,a)


import matplotlib.pyplot as plt
import numpy as np

og_speed = 0.0000001
speed = og_speed
def on_press(event):
    global speed
    if event.key == ' ':
        if speed == og_speed:
            speed = 0.3
        else:
            speed = og_speed

def draw_grid(ax, grid_data, title="RL Grid", color="black"):
    ax.clear()
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold', color=color)
    ax.imshow(grid_data, cmap='Greys', vmin=0, vmax=1)
    
    # Draw Grid Lines
    ax.set_xticks(np.arange(0, cols, 1))
    ax.set_yticks(np.arange(0, rows, 1))
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(False, which='major', axis='both')
    ax.grid(True, which='minor', color='black', linestyle='-', linewidth=2, axis='both')
    ax.tick_params(which='major', top=True, labeltop=True, bottom=False, labelbottom=False)

    for r in range(rows):
        for c in range(cols):
            if grid_data[r, c] == start_marker:
                ax.text(c, r, 'S', ha='center', va='center', fontsize=20, color='blue', fontweight='bold')
            elif grid_data[r, c] == end_marker:
                ax.text(c, r, 'E', ha='center', va='center', fontsize=20, color='green', fontweight='bold')
            elif grid_data[r, c] == reward_marker:
                ax.text(c, r, 'R', ha='center', va='center', fontsize=20, color='orange', fontweight='bold')
    plt.draw()

def draw_policy(policy):
    arrows = ['↑', '↓', '←', '→']
    for r in range(rows):
        for c in range(cols):
            best_action = np.argmax(policy[r, c])
            text_handles[(r, c, 'arrow')].set_text(arrows[best_action])
            
            vals = [f"{i}: {policy[r, c, i]:.2f}" for i in range(4)]
            text_handles[(r, c, 'vals')].set_text("\n".join(vals))
    plt.draw()

def e_greedy_action_selection(agent_idx, state, action_values, eps):
    if np.random.rand() < eps:
        return np.random.choice(4)  # Explore
    else:
        return np.argmax(action_values[agent_idx][state])  # Exploit
    
# 0=Monte Carlo, 1=SARSA, 2=Expected SARSA, 3=Q-learning
methods = ["Monte Carlo", "SARSA", "Expected SARSA", "Q-learning"]
colors = ['red', 'green', 'blue', 'purple']
methods_to_run = [0, 1, 3] # Can run up to 3 methods in parallel for comparison, 
display_policy = len(methods_to_run) == 1
# only displays value function and policy with a single method selected

# special cells on the grid
obstacle_marker = 1 # obstacle cells (impassable) - represented as 1 in the grid for visualization
start_marker = -1 # start cell (agent's initial position)
end_marker = -2 # end cell (goal)
reward_marker = -3 # reward cells (optional)

maze = 2 # 0 for small/easy maze, 1 for medium maze, 2 for large/hard maze
if maze == 2:
    # large/hard maze config
    rows, cols = 7, 11
    grid = np.zeros((rows, cols))
    fs = 4.5 # font size for action values displayed on the grid
    start_pos = (0, 0)
    grid[start_pos] = start_marker
    grid[6, 10] = end_marker
    grid[0:2, 2] = obstacle_marker
    grid[5:7, 2] = obstacle_marker
    grid[1:6, 5] = obstacle_marker
    grid[0, 8] = obstacle_marker
    grid[3:7, 8] = obstacle_marker
    grid[6, 5] = reward_marker
    grid[2, 8] = reward_marker
    episode_length = 60
    total_episodes = 1000
elif maze == 1:
    # medium maze config
    rows, cols = 5, 8
    grid = np.zeros((rows, cols))
    fs = 7 # font size for action values displayed on the grid
    start_pos = (0, 0)
    grid[start_pos] = start_marker
    grid[4, 7] = end_marker
    grid[0:4, 2] = obstacle_marker
    grid[1:5, 5] = obstacle_marker
    episode_length = 40
    total_episodes = 1000
else:
    # small/easy maze config
    rows, cols = 3, 5
    grid = np.zeros((rows, cols))
    fs = 12 # font size for action values displayed on the grid
    start_pos = (2, 0)
    grid[start_pos] = start_marker
    grid[2, 4] = end_marker
    grid[1:3, 2] = obstacle_marker
    episode_length = 20
    total_episodes = 500

# Hyperparameters
gamma = 0.8  # Discount factor
alpha = 0.1  # Learning rate
eps_max = 0.9  # Exploration rate
eps_min = 0.1

# rewards
base_reward = -1  # Base reward for each step
end_reward = 0    # Reward for reaching the end
reward_cell_reward = 1  # Reward for reaching a reward cell (if using)

if len(methods_to_run) == 2:
    layout = """
    AAABBB
    DDEEEF
    """
else:
    layout = """
    AAABBBCCC
    DDDDEEEEF
    """

fig, axs = plt.subplot_mosaic(layout, figsize=(16, 8), facecolor="#C8B29D")

cid = fig.canvas.mpl_connect('key_press_event', on_press)

if display_policy:
    draw_grid(axs['A'], grid, title=methods[methods_to_run[0]], color=colors[methods_to_run[0]])
    draw_grid(axs['B'], np.zeros((rows, cols)), title="Current Policy")
    draw_grid(axs['C'], np.zeros((rows, cols)), title="Action-Value Function")

    # Initialize text handles for policy arrows and action-value values
    text_handles = {}
    for r in range(rows):
        for c in range(cols):
            text_handles[(r, c, 'arrow')] = axs['B'].text(c, r, '', ha='center', va='center', fontsize=fs*5)
            text_handles[(r, c, 'vals')] = axs['C'].text(c, r, '', ha='center', va='center', fontsize=fs)
else:
    assert len(methods_to_run) <= 3, "Can only display up to 3 methods in parallel"

    for i, method in enumerate(methods_to_run):
        draw_grid(axs[chr(65 + i)], grid, title=methods[method], color=colors[method])

axs['D'].axis('off')
info_text = axs['D'].text(0.3, 0.5, f"episode 0\nepsilon (ε) = {eps_max:.2f}", 
                  ha='left', va='center', 
                  fontsize=20, fontweight='bold', color='black',
                  transform=axs['D'].transAxes)
axs['D'].text(0.3, 0.3, f"learning rate (α) = {alpha}\ngamma (γ) = {gamma}", 
            ha='left', va='center', 
            fontsize=15, color='black',
            transform=axs['D'].transAxes)


axs['E'].set_title("Average Reward", pad=20, fontsize=14, fontweight='bold')
axs['E'].set_xlabel("Episode", fontsize=12)
axs['E'].set_ylabel("Average Reward", fontsize=12)
axs['E'].set_xlim(-(total_episodes*0.04), total_episodes*1.04)
axs['E'].set_ylim(-(episode_length*1.04), 1)
axs['E'].grid(True, which='major', color='gray', linestyle='--', linewidth=0.5)
axs['E'].tick_params(axis='both', which='major', width=2.5, length=6)
for spine in axs['E'].spines.values():
    spine.set_linewidth(2.5)

axs['F'].axis('off')

agent_path_lines = []
rewards_lines = []
if display_policy:
    agent_path_line, = axs['A'].plot([], [], color=colors[methods_to_run[0]], linewidth=3, alpha=0.8, zorder=2)
    rewards_line, = axs['E'].plot([], [], label=methods[methods_to_run[0]], color=colors[methods_to_run[0]], linewidth=2, alpha=0.8, zorder=2)
    agent_path_lines.append(agent_path_line)
    rewards_lines.append(rewards_line)
else:
    for i, method in enumerate(methods_to_run):
        agent_path_line, = axs[chr(65 + i)].plot([], [], color=colors[method], linewidth=3, alpha=0.8, zorder=2)
        rewards_line, = axs['E'].plot([], [], label=methods[method], color=colors[method], linewidth=2, alpha=0.8, zorder=2)
        agent_path_lines.append(agent_path_line)
        rewards_lines.append(rewards_line)
axs['E'].legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show(block=False)

episode_rewards = [np.array([[0, -episode_length]]) for _ in methods_to_run] # Initialize with episode 0 and average reward of -episode_length (worst case)
total_reward = [0] * len(methods_to_run) # Total reward accumulated in the current evaluation window for each method
action_value = np.zeros((len(methods_to_run), rows, cols, 4))  # Up, Down, Left, Right
if display_policy:
    draw_policy(action_value[0])
evaluate_every = 10  # Evaluate average reward
for episode in range(total_episodes):
    current_eps = eps_max - (eps_max - eps_min) * (episode / total_episodes)  # Decaying epsilon

    for i, method in enumerate(methods_to_run):
        state = start_pos
        next_action = None
        sample_trajectory = []
        path_x = [start_pos[1]]
        path_y = [start_pos[0]]
        claimed_rewards = set()  # To track reward cells visited in this episode
        for step in range(episode_length):
            r, c = state
            if method == 1 and step != 0:  # SARSA Takes the predetermined next action from the previous step
                action = next_action
            else:
                action = e_greedy_action_selection(i, state, action_value, current_eps)

            # Take action and observe new state and reward
            if action == 0:  # Up
                new_state = (max(r - 1, 0), c)
            elif action == 1:  # Down
                new_state = (min(r + 1, rows - 1), c)
            elif action == 2:  # Left
                new_state = (r, max(c - 1, 0))
            elif action == 3:  # Right
                new_state = (r, min(c + 1, cols - 1))

            if grid[new_state] == obstacle_marker:
                new_state = state

            if grid[new_state] == end_marker:
                reward = end_reward
            elif grid[new_state] == reward_marker and new_state not in claimed_rewards:
                reward = reward_cell_reward  # Give a positive reward for reaching an unclaimed reward cell
                claimed_rewards.add(new_state)  # Mark this reward cell as claimed
            else:
                reward = base_reward

            if method != 0:  # Temporal Difference methods update after every step
                if grid[new_state] == end_marker:
                    expected_next_value = 0  # No future rewards after reaching the end
                elif method == 1:  # SARSA
                    next_action = e_greedy_action_selection(i, state, action_value, current_eps)
                    expected_next_value = action_value[i][new_state][next_action] # Q(St+1​,At+1​)
                elif method == 2:  # Expected SARSA
                    action_probs = np.ones(4) * (current_eps / 4)
                    best_next_action = np.argmax(action_value[i][new_state])
                    action_probs[best_next_action] += (1.0 - current_eps)
                    expected_next_value = np.dot(action_probs, action_value[i][new_state]) # Σa π(a|St+1​) Q(St+1​,a)
                elif method == 3:  # Q-learning
                    expected_next_value = np.max(action_value[i][new_state]) # maxa Q(St+1​,a)

                target = reward + gamma * expected_next_value # Target for TD methods Rt+1 + γ * (next state value)
                action_value[i, r, c, action] += alpha * (target - action_value[i, r, c, action])
            else:
                sample_trajectory.append([state, action, new_state, reward])

            total_reward[i] += reward
            path_x.append(new_state[1])
            path_y.append(new_state[0])
            state = new_state

            if grid[new_state] == end_marker:
                break

        if (episode + 1) % evaluate_every == 0:  
            episode_rewards[i] = np.vstack([episode_rewards[i], [episode + 1, total_reward[i]/evaluate_every]])
            rewards_lines[i].set_data(episode_rewards[i][:, 0], episode_rewards[i][:, 1])
            total_reward[i] = 0

        agent_path_lines[i].set_data(path_x, path_y)

        # Update action-value function using the sampled trajectory
        # Monte Carlo update happens for all state-action pairs in the trajectory after the episode ends
        if method == 0: # Monte Carlo
            action_return = 0.0
            for state, action, new_state, reward in reversed(sample_trajectory):
                # Monte Carlo Return Calculation: Gt = Rt + γRt+1 + γ^2Rt+2 + γ^3Rt+3 + ...
                action_return = reward + gamma * action_return # Gt = Rt + γGt+1
                r, c = state
                # Monte Carlo Update: Q(St​,At​)←Q(St​,At​)+α[Gt​−Q(St​,At​)]
                action_value[i, r, c, action] += alpha * (action_return - action_value[i, r, c, action])

    info_text.set_text(f"episode {episode + 1}\nepsilon (ε) = {current_eps:.2f}")
    if display_policy:
        draw_policy(action_value[0])
    plt.draw()
    plt.pause(speed)

# leave each agents greedy path displayed on the chart at the end of training
for i in range(len(methods_to_run)):
    state = start_pos
    path_x = [start_pos[1]]
    path_y = [start_pos[0]]
    for step in range(episode_length):
        r, c = state
        action = np.argmax(action_value[i][state])

        if action == 0:  # Up
            new_state = (max(r - 1, 0), c)
        elif action == 1:  # Down
            new_state = (min(r + 1, rows - 1), c)
        elif action == 2:  # Left
            new_state = (r, max(c - 1, 0))
        elif action == 3:  # Right
            new_state = (r, min(c + 1, cols - 1))

        if grid[new_state] == obstacle_marker:
            new_state = state

        path_x.append(new_state[1])
        path_y.append(new_state[0])
        state = new_state

        if grid[new_state] == end_marker:
            break
    
    agent_path_lines[i].set_data(path_x, path_y)

plt.draw()
plt.show()