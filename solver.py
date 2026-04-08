import argparse
import copy
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import random

import numpy as np
import torch
import yaml
import magiccube
from tqdm import tqdm

from utils.functions import load_model_class
from train_2x2_heuristic_fc import HeuristicNet


def get_cube_state_str(cube: magiccube.Cube) -> str:
    return str(cube).replace(" ", "").replace("\n", "")

# 2x2 cube moves
MOVES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]
INVERSE_MOVES = {"U": "U'", "U'": "U", "U2": "U2", 
                 "R": "R'", "R'": "R", "R2": "R2",
                 "F": "F'", "F'": "F", "F2": "F2"}
COLORS = {'Y': 0, 'R': 1, 'G': 2, 'O': 3, 'B': 4, 'W': 5}
SOLVED_STATE = get_cube_state_str(magiccube.Cube(2))

def is_solved(cube: magiccube.Cube) -> bool:
    state_str = get_cube_state_str(cube)
    return state_str == SOLVED_STATE


def scramble_cube(num_moves: int = 20) -> Tuple[magiccube.Cube, List[str]]:
    cube = magiccube.Cube(2)
    scramble = []
    
    for _ in range(num_moves):
        move = random.choice(MOVES)
        cube.rotate(move)
        scramble.append(move)
    
    return cube, scramble


@dataclass(order=True)
class SearchNode:
    f_score: float
    g_score: int = field(compare=False)
    state_str: str = field(compare=False)
    node_id: int = field(compare=False)


class SmallNetHeuristic:    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = HeuristicNet(
            input_size=24 * 6,  # 24 positions, one-hot with 6 colors
            hidden_size=512,
            num_layers=4,
            output_size=1
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()

    def _cube_to_onehot(self, cube: magiccube.Cube) -> np.ndarray:
        # Convert cube state to one-hot encoding
        state_str = get_cube_state_str(cube)
        state = [COLORS[c] for c in state_str]
        
        one_hot = np.zeros((24, 6), dtype=np.float32)
        for i, color in enumerate(state):
            one_hot[i, color] = 1.0
        return one_hot.flatten()
    
    @torch.no_grad()
    def __call__(self, cube: magiccube.Cube) -> float:
        return self.batch_call([cube])[0]
    
    @torch.no_grad()
    def batch_call(self, cubes: List[magiccube.Cube]) -> List[float]:
        if not cubes:
            return []
        
        # Convert cubes to one-hot tensors
        inputs = np.stack([self._cube_to_onehot(cube) for cube in cubes])
        inputs = torch.from_numpy(inputs).to(self.device)
        
        # Run model
        outputs = self.model(inputs)
        heuristics = outputs.cpu().tolist()
        
        return heuristics


def astar_solve(
    cube: magiccube.Cube,
    heuristic_fn,
    max_nodes: int = 100000,
    batch_size: int = 512
):
    metrics = {
        "nodes_expanded": 0,
        "nodes_generated": 0,
        "max_queue_size": 0,
        "solution_length": None,
        "time_seconds": 0,
        "heuristic_calls": 0,
    }
    
    start_time = time.time()
    
    if is_solved(cube):
        metrics["solution_length"] = 0
        return [], metrics
    
    cubes = {}
    parent_map = {}  # node_id -> (parent_id, move)
    node_counter = [0]
    
    def get_next_node_id():
        node_counter[0] += 1
        return node_counter[0]
    
    def reconstruct_path(node_id):
        path = []
        current_id = node_id
        while current_id in parent_map:
            parent_id, move = parent_map[current_id]
            path.append(move)
            current_id = parent_id
        return list(reversed(path))
    
    # Initial heuristic
    initial_h = heuristic_fn(cube)
    metrics["heuristic_calls"] += 1
    
    start_state = get_cube_state_str(cube)
    start_node_id = get_next_node_id()
    cubes[start_node_id] = copy.deepcopy(cube)
    
    start_node = SearchNode(
        f_score=initial_h,
        g_score=0,
        state_str=start_state,
        node_id=start_node_id
    )
    
    # Priority queue and visited set
    open_set = [start_node]
    visited = {start_state: 0}  # state_str -> best g_score
    
    pbar = tqdm(total=max_nodes, desc="A* Search", unit=" nodes")
    
    while open_set and metrics["nodes_expanded"] < max_nodes:
        metrics["max_queue_size"] = max(metrics["max_queue_size"], len(open_set))
        
        # Pop multiple nodes to expand in batch
        nodes_to_expand = []
        while open_set and len(nodes_to_expand) < max(1, batch_size // len(MOVES)): # ensure expanded nodes are <= than batch size
            node = heapq.heappop(open_set)
            # Skip if there is a better path to this state
            if node.g_score <= visited.get(node.state_str, float('inf')):
                nodes_to_expand.append(node)
        
        if not nodes_to_expand:
            break
        
        nodes_expanded_before = metrics["nodes_expanded"]
        metrics["nodes_expanded"] += len(nodes_to_expand)
        pbar.update(metrics["nodes_expanded"] - nodes_expanded_before)
        
        # Generate successor states for all nodes
        all_new_cubes = []
        all_new_states = []
        all_new_info = []  # (parent_node_id, move, new_g)
        
        for current in nodes_to_expand:
            new_g = current.g_score + 1
            current_cube = cubes[current.node_id]
            
            for move in MOVES:
                # Create new cube
                new_cube = copy.deepcopy(current_cube)
                new_cube.rotate(move)
                new_state = get_cube_state_str(new_cube)
                
                metrics["nodes_generated"] += 1
                
                # Check if solved
                if new_state == SOLVED_STATE:
                    solution = reconstruct_path(current.node_id) + [move]
                    metrics["solution_length"] = len(solution)
                    metrics["time_seconds"] = time.time() - start_time
                    pbar.close()
                    return solution, metrics
                
                # Skip if there is a better path to this state
                if new_state in visited and visited[new_state] <= new_g:
                    continue
                
                visited[new_state] = new_g
                all_new_cubes.append(new_cube)
                all_new_states.append(new_state)
                all_new_info.append((current.node_id, move, new_g))
        
        # Batch compute heuristics for all successors
        if all_new_cubes:
            heuristics = heuristic_fn.batch_call(all_new_cubes)
            metrics["heuristic_calls"] += len(all_new_cubes)
            
            for new_cube, new_state, (parent_id, move, new_g), h in zip(
                all_new_cubes, all_new_states, all_new_info, heuristics
            ):
                new_node_id = get_next_node_id()
                cubes[new_node_id] = new_cube
                parent_map[new_node_id] = (parent_id, move)
                
                new_node = SearchNode(
                    f_score=new_g + h,
                    g_score=new_g,
                    state_str=new_state,
                    node_id=new_node_id
                )
                heapq.heappush(open_set, new_node)

    pbar.close()
    
    metrics["time_seconds"] = time.time() - start_time
    return None, metrics


class ZeroHeuristic:    
    def __call__(self, cube: magiccube.Cube) -> float:
        return 0.0
    
    def batch_call(self, cubes: List[magiccube.Cube]) -> List[float]:
        return [0.0] * len(cubes)


def run_evaluation(
    heuristic_fn,
    num_cubes: int = 10,
    scramble_moves: int = 20,
    max_nodes: int = 100000,
    batch_size: int = 512,
    heuristic_name: str = "neural"
):
    print(f"Evaluating {heuristic_name} heuristic on {num_cubes} cubes")
    print(f"Scramble moves: {scramble_moves}, Max nodes: {max_nodes}, Batch size: {batch_size}")
    
    results = []
    
    for i in range(num_cubes):
        cube, scramble = scramble_cube(scramble_moves)
        print(f"Cube {i+1}/{num_cubes}: scramble = {' '.join(scramble[:10])}...")
        
        solution, metrics = astar_solve(cube, heuristic_fn, max_nodes, batch_size)
        
        if solution is not None:
            print(f"  Solved in {metrics['solution_length']} moves")
            print(f"  Nodes: {metrics['nodes_expanded']}, Time: {metrics['time_seconds']:.3f}s")
        else:
            print(f"  Not solved (max nodes reached)")
        
        results.append({
            "scramble": scramble,
            "solution": solution,
            **metrics
        })
    
    # Summary statistics
    solved = [r for r in results if r["solution"] is not None]
    
    print(f"Solved: {len(solved)}/{num_cubes} ({100*len(solved)/num_cubes:.1f}%)")
    
    if solved:
        avg_length = sum(r["solution_length"] for r in solved) / len(solved)
        avg_nodes = sum(r["nodes_expanded"] for r in solved) / len(solved)
        avg_time = sum(r["time_seconds"] for r in solved) / len(solved)
        avg_heuristic_calls = sum(r["heuristic_calls"] for r in solved) / len(solved)
        
        print(f"Avg solution length: {avg_length:.2f}")
        print(f"Avg nodes expanded: {avg_nodes:.1f}")
        print(f"Avg time: {avg_time:.3f}s")
        print(f"Avg heuristic calls: {avg_heuristic_calls:.1f}")


def main():
    parser = argparse.ArgumentParser(description="2x2 Cube Solver with Neural Heuristic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="smallnet", choices=["smallnet", "zero"])
    parser.add_argument("--scramble_moves", type=int, default=20)
    parser.add_argument("--num_cubes", type=int, default=10)
    parser.add_argument("--max_nodes", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load heuristic based on model type
    if args.model_type == "smallnet":
        heuristic = SmallNetHeuristic(args.checkpoint)
        heuristic_name = "SmallNet"
    else:
        heuristic = ZeroHeuristic()
        heuristic_name = "Zero (BFS)"
    
    # Run evaluation
    run_evaluation(
        heuristic,
        num_cubes=args.num_cubes,
        scramble_moves=args.scramble_moves,
        max_nodes=args.max_nodes,
        batch_size=args.batch_size,
        heuristic_name=heuristic_name
    )


if __name__ == "__main__":
    main()

