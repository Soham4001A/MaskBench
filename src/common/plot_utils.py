import matplotlib.pyplot as plt
import numpy as np

def render_sweep_plot(all_rewards, out_path, title):
    plt.figure(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_rewards)))
    
    negative_trending = sum(1 for r in all_rewards.values() if np.sum(r) < 0) > len(all_rewards) / 2

    for i, (prob, rewards) in enumerate(all_rewards.items()):
        linestyle = 'dotted' if prob == 0.0 else '-'
        plt.plot(np.cumsum(rewards), label=f'p={prob}', color=colors[i], linestyle=linestyle)
    
    if negative_trending:
        plt.gca().invert_yaxis()

    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved sweep plot to {out_path}")
