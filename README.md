# 🔷 N-in-a-Row Adversarial Search (Minimax & Alpha-Beta Pruning)

This project explores adversarial search in **N-in-a-Row**, a generalised version of 
Connect-Four. We implemented and compared two classic game tree algorithms:

- **Depth-Limited Minimax**
- **Minimax with Alpha–Beta Pruning**

The goal was to analyse search efficiency through the number of **evaluated board 
states**, providing a hardware-independent measure of computational complexity.

---

## 🎮 What the Project Does

- Represents N-in-a-Row game states programmatically
- Implements Minimax for optimal adversarial decision-making
- Adds Alpha–Beta pruning to reduce unnecessary search
- Runs experiments to compare search efficiency
- Reports performance across different **depths and N-values**

---

## 🧪 Key Experimental Results

- Tested on a **7×6 grid**
- **N ∈ {3, 4}**
- Depths **2–5**

Alpha–Beta pruning consistently evaluated **significantly fewer states** than the 
baseline Minimax algorithm — with efficiency gains increasing at deeper search levels.

> **Conclusion:** Pruning enables dramatic runtime reduction while preserving 
optimal play — reinforcing classical game-search theory.



