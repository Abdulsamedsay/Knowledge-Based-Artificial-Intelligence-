from __future__ import annotations
import time, random
from typing import List, Tuple, Dict

from heuristics import SimpleHeuristic
from players import MinMaxPlayer, AlphaBetaPlayer
from board import Board


#Abdulsamed Say (s1146476)

def random_midgame(board: Board, game_n: int, plies: int, seed: int = 0) -> Board:
    rng = random.Random(seed)
    cur = Board(board.width, board.height)
    player = 1
    for _ in range(plies):
        legal = [c for c in range(cur.width) if cur.is_valid(c)]
        if not legal:
            break
        col = rng.choice(legal)
        cur = cur.get_new_board(col, player)
        # stop if terminal
        h = SimpleHeuristic(game_n)
        if h.winning(cur.get_board_state(), game_n) != 0:
            break
        player = 2 if player == 1 else 1
    return cur

def measure_one_decision(state: Board, game_n: int, depth: int) -> Dict[str, int]:
    h_min = SimpleHeuristic(game_n)
    h_ab  = SimpleHeuristic(game_n)

    p_min = MinMaxPlayer(1, game_n, depth=depth, heuristic=h_min)
    p_ab  = AlphaBetaPlayer(1, game_n, depth=depth, heuristic=h_ab)

    # MINIMAX
    h_min.eval_count = 0
    t0 = time.perf_counter_ns()
    _ = p_min.make_move(Board(state.get_board_state()))
    t1 = time.perf_counter_ns()

    # ALPHA-BETA on the exact same state
    h_ab.eval_count = 0
    t2 = time.perf_counter_ns()
    _ = p_ab.make_move(Board(state.get_board_state()))
    t3 = time.perf_counter_ns()

    return dict(
        nodes_minimax=h_min.eval_count,
        nodes_alphabeta=h_ab.eval_count,
        time_minimax_ns=(t1 - t0),
        time_alphabeta_ns=(t3 - t2),
    )


def one_match(depth: int, width: int = 7, height: int = 6, game_n: int = 4) -> Tuple[int, int]:
    """
    Legacy: runs a *full game* Minimax (X) vs AlphaBeta (O), returns total eval counts.
    Keep it if you want to show match totals in the video, but don't use it for the main comparison.
    """
    from app import start_game
    board = Board(width, height)
    h1 = SimpleHeuristic(game_n)
    h2 = SimpleHeuristic(game_n)
    p1 = MinMaxPlayer(1, game_n, depth=depth, heuristic=h1)      # X = Minimax
    p2 = AlphaBetaPlayer(2, game_n, depth=depth, heuristic=h2)   # O = Alpha–Beta
    start_game(game_n, board, [p1, p2])
    return h1.eval_count, h2.eval_count

def sweep_experiments(
    depths: List[int] = [2,3,4,5],
    sizes: List[Tuple[int,int]] = [(7,6), (8,7)],
    Ns: List[int] = [3,4],
    seeds: int = 8,
    prefill_moves: int = 10
) -> List[Dict]:
    rows = []
    for (W,H) in sizes:
        for N in Ns:
            for d in depths:
                for s in range(seeds):
                    base = Board(W, H)
                    state = random_midgame(base, N, plies=prefill_moves, seed=10_000 + s)
                    res = measure_one_decision(state, game_n=N, depth=d)
                    rows.append(dict(
                        width=W, height=H, N=N, depth=d, seed=s,
                        **res
                    ))
    return rows

if __name__ == "__main__":
    for d in [2, 3, 4, 5]:
        m, a = one_match(d, width=7, height=6, game_n=3)
        print(f"[MATCH TOTAL] DEPTH {d}: Minimax={m}  AlphaBeta={a}")

    print("width,height,N,depth,seed,nodes_minimax,nodes_alphabeta,time_minimax_ns,time_alphabeta_ns")
    for row in sweep_experiments():
        print("{width},{height},{N},{depth},{seed},{nodes_minimax},{nodes_alphabeta},{time_minimax_ns},{time_alphabeta_ns}".format(**row))

