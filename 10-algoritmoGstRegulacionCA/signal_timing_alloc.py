
#!/usr/bin/env python3
import argparse
import math
from typing import List, Optional

def parse_list(s: str) -> List[float]:
    # Accept comma-separated like "500,350,250"
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def allocate_green_times(
    cycle_s: float,
    lost_per_phase_s: float,
    volumes: List[float],
    spacings: Optional[List[float]] = None,
    ped_start_s: Optional[float] = None,
    ped_cross_s: Optional[float] = None,
    yellow_s: Optional[float] = None,
    round_to: float = 0.1
) -> dict:
    n = len(volumes)
    if n == 0:
        raise ValueError("Provide at least one volume value.")
    if spacings is not None and len(spacings) != n:
        raise ValueError("If spacings are provided, they must match the number of volumes.")

    total_lost = lost_per_phase_s * n
    green_available = cycle_s - total_lost
    if green_available <= 0:
        raise ValueError("Green available is non-positive. Reduce losses or increase cycle.")

    # Weights: volumes or volumes * spacings
    if spacings is None:
        weights = volumes[:]
        mode = "volumes_only"
    else:
        weights = [v * e for v, e in zip(volumes, spacings)]
        mode = "volume_times_spacing"

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Sum of weights must be positive.")

    # Initial proportional allocation
    greens = [green_available * w / weight_sum for w in weights]

    # Compute pedestrian minimum (optional)
    ped_min = None
    if ped_start_s is not None and ped_cross_s is not None and yellow_s is not None:
        ped_min = ped_start_s + ped_cross_s - yellow_s
        ped_min = max(0.0, ped_min)

    # Enforce minimums if any (e.g., pedestrian minimum)
    if ped_min is not None:
        # Raise greens below ped_min and re-normalize remaining
        # Step 1: lock phases that need minimum
        locked = [max(g, ped_min) for g in greens]
        locked_sum = sum(locked)
        if locked_sum > green_available:
            # Not feasible: set all to ped_min and scale down uniformly
            scale = green_available / (ped_min * n) if ped_min > 0 else 0.0
            greens = [ped_min * scale for _ in range(n)]
        else:
            # Distribute remaining among phases that were above min proportionally to their original share
            remaining = green_available - sum(max(ped_min, g) for g in greens)
            # Build proportional shares only for phases originally above ped_min
            # If none were above ped_min, just keep all at ped_min (already handled by locked_sum check)
            above = [i for i, g in enumerate(greens) if g > ped_min]
            if above:
                total_above = sum(greens[i] - ped_min for i in above)
                new_greens = []
                for i, g in enumerate(greens):
                    if g > ped_min and total_above > 0:
                        extra = (g - ped_min) / total_above * remaining
                        new_greens.append(ped_min + extra)
                    else:
                        new_greens.append(ped_min)
                greens = new_greens
            else:
                greens = [ped_min for _ in greens]

    # Round to the desired resolution and rebalance to exact total
    if round_to and round_to > 0:
        greens = [round(g / round_to) * round_to for g in greens]
        # Rebalance to match green_available exactly (small corrections)
        diff = green_available - sum(greens)
        # Distribute the diff in steps of round_to
        step = round_to if diff >= 0 else -round_to
        i = 0
        while abs(diff) >= round_to - 1e-9 and i < 10000:
            greens[i % n] += step
            diff -= step
            i += 1

    return {
        "mode": mode,
        "cycle_s": cycle_s,
        "lost_per_phase_s": lost_per_phase_s,
        "n_phases": n,
        "green_available_s": green_available,
        "volumes_vph": volumes,
        "spacings_s": spacings,
        "ped_min_s": ped_min,
        "greens_s": greens,
        "check_sum_s": sum(greens)
    }

def main():
    p = argparse.ArgumentParser(description="Allocate green times for N phases using proportional methods.")
    p.add_argument("--cycle", type=float, required=True, help="Cycle length (s)")
    p.add_argument("--lost_per_phase", type=float, required=True, help="Lost time per phase (s)")
    p.add_argument("--volumes", type=str, required=True, help="Comma-separated volumes, e.g. 500,350,250")
    p.add_argument("--spacings", type=str, default=None, help="Comma-separated spacings (s/veh), e.g. 3,4,6 (optional)")
    p.add_argument("--ped_start", type=float, default=None, help="Pedestrian start-walk time (s)")
    p.add_argument("--ped_cross", type=float, default=None, help="Pedestrian crossing time (s)")
    p.add_argument("--yellow", type=float, default=None, help="Yellow interval (s)")
    p.add_argument("--round_to", type=float, default=0.1, help="Rounding resolution (s), default 0.1")

    args = p.parse_args()
    vols = parse_list(args.volumes)
    spcs = parse_list(args.spacings) if args.spacings is not None else None

    result = allocate_green_times(
        cycle_s=args.cycle,
        lost_per_phase_s=args.lost_per_phase,
        volumes=vols,
        spacings=spcs,
        ped_start_s=args.ped_start,
        ped_cross_s=args.ped_cross,
        yellow_s=args.yellow,
        round_to=args.round_to
    )

    # Pretty print
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
