import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6), facecolor="grey")

models = ["ll_dqn1", "ll_pg1", "ll_ppo1", "llcont_ppo1", "walk_ppo1", "llcont_sac1", "llcont_sac2", "walk_sac1", "walk_sac2", "human_sac1"]
colors = ['indigo', 'g', "#8A2BE2", 'b', 'c', 'g', 'y', 'orange', 'r', 'darkred']
show = [9]

window = 1000
for i in show:
    df = pd.read_csv(f"metrics/{models[i]}_metrics.csv")

    df['rolling_avg'] = df['reward'].rolling(window=window, center=True).mean()

    plt.plot(df['step'], df['reward'], alpha=0.1, color=colors[i])
    plt.plot(df['step'], df['rolling_avg'], color=colors[i], linewidth=2, label=models[i])

plt.xlabel('Time')
plt.ylabel('Total Reward')
plt.title('Total reward per Episode')
#plt.ylim(-350, 350)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()