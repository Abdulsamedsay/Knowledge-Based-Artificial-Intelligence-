import pandas as pd
import matplotlib.pyplot as plt

#Abdulsamed Say (s1146476)
#Ismail Vatansever (s1152889)

df = pd.read_csv("results_minimax_vs_alphabeta.csv")

# Group by width, height, N, depth
summary = df.groupby(["width","height","N","depth"]).agg(
    nodes_min_mean=("nodes_minimax","mean"),
    nodes_ab_mean=("nodes_alphabeta","mean"),
    time_min_mean=("time_minimax_ns","mean"),
    time_ab_mean=("time_alphabeta_ns","mean"),
).reset_index()

print(summary)



for (w,h,N), sub in summary.groupby(["width","height","N"]):
    plt.figure()
    plt.plot(sub["depth"], sub["nodes_min_mean"], label="Minimax")
    plt.plot(sub["depth"], sub["nodes_ab_mean"], label="Alpha-Beta")
    plt.xlabel("Search depth")
    plt.ylabel("Nodes evaluated (avg)")
    plt.title(f"{w}x{h} board, N={N}")
    plt.legend()
    plt.savefig(f"nodes_vs_depth_{w}x{h}_N{N}.png")
