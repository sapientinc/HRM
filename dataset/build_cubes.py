import json
import os
import sys
from argdantic import ArgParser
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import py222

from common import PuzzleDatasetMetadata
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading
import time
import os

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/cube-2-by-2"

    seed: int = 42
    train_size: int = 100000
    test_size: int = 300
    validation_size: int = 300
    
    min_scramble_moves: int = 5
    max_scramble_moves: int = 20


def scramble_cube(rng, min_moves, max_moves):
    s = py222.initState()
    n_moves = rng.integers(min_moves, max_moves + 1)
    
    last_move_face = -1  # avoid moving twice the same side
    for _ in range(n_moves):
        while True:
            move = rng.integers(0, 9)  # we only need U R F since for the 2x2 F is the same as B'
            move_face = move // 3  # 0=U, 1=R, 2=F
            if move_face != last_move_face:
                break
        s = py222.doMove(s, move)
        last_move_face = move_face
    
    return s


def generate_sample(rng, min_scramble, max_scramble):
    scrambled = scramble_cube(rng, min_scramble, max_scramble)
    solution = py222.solve(scrambled)
    return scrambled, solution


def _init_worker():
    import py222


def _process_worker_batch(seed, min_scramble, max_scramble, batch, progress_queue=None):
    local_rng = np.random.default_rng(int(seed))
    items = []
    start_t = time.time()
    for _ in range(batch):
        scrambled, solution = generate_sample(local_rng, min_scramble, max_scramble)
        items.append((scrambled, solution, time.time()))

        if progress_queue is not None:
            progress_queue.put(1)

    sample_times = []
    prev = start_t
    for (_, _, t) in items:
        sample_times.append(t - prev)
        prev = t
    return [(s, sol, dt) for (s, sol, _), dt in zip(items, sample_times)], os.getpid()


def create_dataset(set_name, size, config: DataProcessConfig):
    # train, test, val should not have same seed
    if set_name == "train":
        seed = config.seed
    elif set_name == "test":
        seed = config.seed + 1000
    else:  # val
        seed = config.seed + 2000
    
    rng = np.random.default_rng(seed)
    
    seq_length = 24  # state string has 24 chars, larger than max moves 11
    
    # Generate cubes
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }
    
    puzzle_id = 0
    example_id = 0
    
    # Parallel sample generation
    max_workers = os.cpu_count() or 1

    batch_size = max(1, size // (max_workers * 4))
    num_jobs = (size + batch_size - 1) // batch_size

    manager = multiprocessing.Manager()
    progress_q = manager.Queue()

    def _progress_reader(q, total):
        pbar = tqdm(total=total, desc=f"Generating {set_name}")
        while True:
            msg = q.get()
            if msg is None:
                break
            pbar.update(msg)
        pbar.close()

    reader_thread = threading.Thread(target=_progress_reader, args=(progress_q, size), daemon=True)
    reader_thread.start()

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as exe:
        # create seeds for jobs
        job_seeds = [int(x) for x in rng.integers(0, 2**32 - 1, size=num_jobs, dtype=np.uint64).tolist()]
        futures = [exe.submit(_process_worker_batch, seed, config.min_scramble_moves, config.max_scramble_moves, batch_size if i < num_jobs - 1 else (size - i * batch_size), progress_q) for i, seed in enumerate(job_seeds)]

        elapsed_times = []
        pid_counts = {}
        pid_times = {}
        completed = 0
        for fut in tqdm(as_completed(futures), total=len(futures)):
            batch_items, pid = fut.result()
            pid_counts[pid] = pid_counts.get(pid, 0) + len(batch_items)
            pid_times.setdefault(pid, []).extend([t for (_, _, t) in batch_items])

            for scrambled, solution, elapsed in batch_items:
                if scrambled is None:
                    continue
                elapsed_times.append(elapsed)

                inp = scrambled + 1  # add padding

                label = np.zeros(seq_length, dtype=np.int32)  # pad labels to seq length
                for i, move in enumerate(solution):
                    if i >= seq_length:
                        break
                    label[i] = move + 1  # add padding

                results["inputs"].append(inp.astype(np.int32))
                results["labels"].append(label)

                example_id += 1
                puzzle_id += 1

                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(0)

                # Push group
                results["group_indices"].append(puzzle_id)

                completed += 1
                if completed >= size:
                    break

        if elapsed_times:
            avg = sum(elapsed_times) / len(elapsed_times)
            print(f"Solve time: avg={avg:.3f}s min={min(elapsed_times):.3f}s max={max(elapsed_times):.3f}s")
            print("Sample counts (pid:count):")
            for pid, cnt in sorted(pid_counts.items(), key=lambda x: -x[1])[:10]:
                times = pid_times.get(pid, [])
                print(f"  {pid}:{cnt} avg={sum(times)/len(times):.3f}s")

        # stop reader thread
        try:
            progress_q.put(None)
        except Exception:
            pass
        reader_thread.join()

    # Convert to numpy arrays
    final_results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    # vocab_size:
    #   input: 6 colors
    #   output: 2x2 cube has only 9 moves: U, U', U2, R, R', R2, F, F', F2
    num_moves = 9
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_length,  # must match input length
        vocab_size=num_moves + 1,  # equals the larger vocab + 1 for pad
        pad_id=0,
        ignore_label_id=0,
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(final_results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    create_dataset("train", config.train_size, config)
    create_dataset("test", config.test_size, config)
    create_dataset("val", config.validation_size, config)

if __name__ == "__main__":
    cli()
