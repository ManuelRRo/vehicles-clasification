from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Deque


# ---------- Data structures ----------

@dataclass
class RoadStats:
    vehicles: int
    emergency_vehicles: int


# ---------- Low-level operations (stubs to replace with your code) ----------

def capture_frames_of_each_road(capture_interval: float):
    """
    Capture frames for each road after 'capture_interval' seconds.
    Return any structure you like, e.g. {road_id: list_of_frames}.
    """
    # TODO: integrate with your cameras / video streams
    frames_by_road = {}
    return frames_by_road


def detect_and_count_vehicles(frames_by_road) -> Dict[int, RoadStats]:
    """
    Detect vehicles and emergency vehicles from frames.
    Returns roads_vehicle dictionary: {road_id: RoadStats(...)}.
    """
    roads_vehicle: Dict[int, RoadStats] = {}

    # TODO: run YOLO or your detector here
    # Example dummy data:
    roads_vehicle[0] = RoadStats(vehicles=10, emergency_vehicles=0)
    roads_vehicle[1] = RoadStats(vehicles=3, emergency_vehicles=1)
    roads_vehicle[2] = RoadStats(vehicles=5, emergency_vehicles=0)
    roads_vehicle[3] = RoadStats(vehicles=1, emergency_vehicles=0)

    return roads_vehicle


def compute_adjusted_green_time(road_id: int, stats: RoadStats) -> float:
    """
    Calculate the adjusted green time for a road.
    Replace the formula with whatever you need.
    """
    base_time = 10.0           # seconds
    per_vehicle = 1.0          # extra seconds per vehicle
    emergency_bonus = 10.0     # extra seconds if there is an emergency vehicle

    t = base_time + per_vehicle * stats.vehicles
    if stats.emergency_vehicles > 0:
        t += emergency_bonus
    return t


def provide_green_light(road_id: int, green_time: float):
    """
    Activate green light for 'road_id' during 'green_time' seconds.
    """
    print(f"Road {road_id}: GREEN for {green_time:.1f} s")
    # TODO: send command to your traffic controller hardware


# ---------- Main control loop ----------
def traffic_controller_main_loop():
    capture_interval = 10.0  # initial interval (seconds)

    previous_sorted_list: List[int] = []
    roads_cycle: Deque[int] = deque()

    while True:
        # 1) Capture frames after capture_interval
        frames_by_road = capture_frames_of_each_road(capture_interval)

        # 2) Detect and count vehicles & emergency vehicles
        roads_vehicle = detect_and_count_vehicles(frames_by_road)

        # 3) roads_vehicle is already our dictionary

        # 4) Sort using emergency first, then number of vehicles (descending)
        sorted_list = sorted(
            roads_vehicle.keys(),
            key=lambda r: (
                roads_vehicle[r].emergency_vehicles > 0,  # True > False
                roads_vehicle[r].vehicles
            ),
            reverse=True
        )

        # 5) If updated sorted list != previous sorted list, update it
        if sorted_list != previous_sorted_list:
            previous_sorted_list = list(sorted_list)

        # 6) For each road in the sorted list
        for road_id in sorted_list:
            # If road not in roads_cycle queue, add it at the end
            if road_id not in roads_cycle:
                roads_cycle.append(road_id)
            # If it *is* already in the queue, just continue to next road
            # (equivalent to "continue from next road" in the diagram)

        # 7) If roads_cycle is empty (safety fallback), fill it with sorted_list
        if not roads_cycle:
            roads_cycle.extend(sorted_list)

        # 8) Select the next road from roads_cycle queue
        current_road = roads_cycle[0]  # look at head of queue

        # 9) Calculate adjusted time of green light for the road
        stats = roads_vehicle[current_road]
        adjusted_time = compute_adjusted_green_time(current_road, stats)

        # 10) Provide green light for adjusted time
        provide_green_light(current_road, adjusted_time)

        # 11) Remove the selected road from roads_cycle queue
        roads_cycle.popleft()

        # 12) Set capture_interval as adjusted_time (for the next iteration)
        capture_interval = adjusted_time


# if __name__ == "__main__":
#     traffic_controller_main_loop()
