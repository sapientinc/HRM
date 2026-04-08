"""
Colorful step-by-step cube solver visualization.

This script:
  1. Takes a scrambled cube state
  2. Solves it with A* using the learned heuristic
  3. Displays each step of the solution with colorful visualization
"""
import numpy as np
import torch
import sys
import copy
from pathlib import Path
from typing import List, Tuple
import magiccube

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from solver import SmallNetHeuristic, ZeroHeuristic, astar_solve, scramble_cube, is_solved


class CubeVisualizer:
    """Visualize 2x2x2 cube states."""
    
    COLOR_SYMBOLS = {
        'W': "⬜",
        'Y': "🟨",
        'O': "🟧",
        'R': "🟥",
        'G': "🟩",
        'B': "🟦",
    }
    
    @staticmethod
    def print_cube_state(cube: magiccube.Cube, title: str = ""):
        """Print cube state in unfolded format."""
        if title:
            print(f"\n{title}")
        
        s = str(cube).replace(" ", "").replace("\n", "")
        
        c = CubeVisualizer.COLOR_SYMBOLS
        
        # Top face (U)
        print(f"       {c[s[0]]} {c[s[1]]}")
        print(f"       {c[s[2]]} {c[s[3]]}")
        print()
        # Middle rows: L F R B
        print(f"{c[s[4]]} {c[s[5]]}  {c[s[6]]} {c[s[7]]}  {c[s[8]]} {c[s[9]]}  {c[s[10]]} {c[s[11]]}")
        print(f"{c[s[12]]} {c[s[13]]}  {c[s[14]]} {c[s[15]]}  {c[s[16]]} {c[s[17]]}  {c[s[18]]} {c[s[19]]}")
        print()
        # Bottom face (D)
        print(f"       {c[s[20]]} {c[s[21]]}")
        print(f"       {c[s[22]]} {c[s[23]]}")
    
    @staticmethod
    def apply_moves_trace(
        cube: magiccube.Cube,
        solution: List[str]
    ) -> List[Tuple[magiccube.Cube, str]]:
        """Apply moves step-by-step and return states + move names."""
        trace = [(copy.deepcopy(cube), "Initial State")]
        
        try:
            current_cube = copy.deepcopy(cube)
            for move in solution:
                current_cube.rotate(move)
                trace.append((copy.deepcopy(current_cube), move))
            return trace
        except Exception as e:
            print(f"Error during move application: {e}")
            return trace


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Settings
    SCRAMBLE_LENGTH = 20
    MAX_NODES = 100000
    
    # Load heuristic model
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "2x2-heuristic-fc" / "100000_examples"
    checkpoint_path = checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        checkpoint_path = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))[-1]
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found in {checkpoint_dir}")
        sys.exit(1)
    
    print(f"Loading heuristic model from {checkpoint_path}...\n")
    heuristic = SmallNetHeuristic(str(checkpoint_path), device=device)
    #heuristic = ZeroHeuristic()
    
    # Generate scramble
    test_cube, scramble = scramble_cube(num_moves=SCRAMBLE_LENGTH)
    print(f"\nScramble ({len(scramble)} moves): {' '.join(scramble)}")
    CubeVisualizer.print_cube_state(test_cube, "Scrambled Cube State")
    
    # Solve with A*
    print(f"\nSolving with A* (max {MAX_NODES} nodes)...")
    
    solution, metrics = astar_solve(test_cube, heuristic, max_nodes=MAX_NODES, batch_size=512)
    
    if solution is not None:
        print(f"\nSolution ({len(solution)} moves): {' '.join(solution)}")
        print(f"Stats: {metrics['nodes_expanded']} nodes, {metrics['time_seconds']:.2f}s")
        
        # Validate solution
        test_cube_copy = copy.deepcopy(test_cube)
        for move in solution:
            test_cube_copy.rotate(move)
        
        is_valid = is_solved(test_cube_copy)
        if not is_valid:
            print("Warning: Solution does not solve the cube!")
        
        # Step-by-step visualization
        print("\nStep-by-step solution:")
        trace = CubeVisualizer.apply_moves_trace(test_cube, solution)
        
        for i, (state, move) in enumerate(trace):
            if i == 0:
                CubeVisualizer.print_cube_state(state, f"Step {i}: {move}")
            else:
                CubeVisualizer.print_cube_state(state, f"Step {i}: {move}")
            print()
    
    else:
        print(f"\nNo solution found within {MAX_NODES} nodes")
        print(f"Nodes expanded: {metrics['nodes_expanded']}, Time: {metrics['time_seconds']:.2f}s")
