import os

def find_last_checkpoint(checkpoint_dir):
    """
    Find the last (most recent) numeric checkpoint in the given directory.

    Args:
        checkpoint_dir (str): The directory containing the checkpoints.

    Returns:
        str or None: The path to the last numeric checkpoint file.
    """
    files = os.listdir(checkpoint_dir)

    checkpoints = []
    for f in files:
        if f.endswith(".pt"):
            try:
                step = int(f.replace(".pt", ""))
                checkpoints.append((step, f))
            except ValueError:
                continue  # skip non-numeric filenames like 'best.pt'

    if not checkpoints:
        return None

    checkpoints.sort()
    return os.path.join(checkpoint_dir, checkpoints[-1][1])


def find_best_checkpoint(checkpoint_dir):
    """
    Find the best checkpoint (named 'best.pt') in the given directory.

    Args:
        checkpoint_dir (str): The directory containing the checkpoints.

    Returns:
        str or None: The path to best.pt if it exists.
    """
    best_path = os.path.join(checkpoint_dir, "best.pt")
    return best_path if os.path.isfile(best_path) else None