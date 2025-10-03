import numpy as np

def to_red_str(val: float) -> str:
    return f"\x1b[31m{val:.3f}\x1b[0m"

def mask_table(obs, masked_obs):
    """
    Returns a list of strings representing a table where masked entries are colored red.
    For vector observations only; for higher dims, shows flattened slice.
    """
    obs = np.asarray(obs)
    masked_obs = np.asarray(masked_obs)
    flat_o = obs.flatten()
    flat_m = masked_obs.flatten()
    rows = []
    for i, (o, m) in enumerate(zip(flat_o, flat_m)):
        if o != m:
            rows.append(f"{i:4d}: {to_red_str(m)} (orig {o:.3f})")
        else:
            rows.append(f"{i:4d}: {m:.3f}")
    return rows[:64]  # limit for readability
