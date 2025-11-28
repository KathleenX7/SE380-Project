import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from simulator import RaceTrack

# Global Variables 
prev_vr_e = 0
vr_integral = 0
prev_delta_e = 0
delta_e_integral = 0
delta_cmd_prev = 0.0
prev_delta_r = 0.0
t_r = 0.01
t_d = 0.1
max_idx = 0
finished = False
last_lookahead = 2

# Velocity PID
Kp_a, Ki_a, Kd_a = 0.75, 0.1, 0.01
max_integral_v = 3.0

# Steering PID
Kp_delta, Ki_delta, Kd_delta = 2.0, 0.05, 0.75
max_integral_delta = 2.0

def lower_controller(state, desired, parameters):
    global vr_integral, prev_vr_e, prev_delta_e, delta_e_integral, delta_cmd_prev

    delta_desired, v_desired = desired
    _, _, delta, v, _ = state

    vr_e = v_desired - v
    delta_e = delta_desired - delta

    # Velocity PID
    vr_integral = np.clip(vr_integral + Ki_a * vr_e * t_r, -max_integral_v, max_integral_v)
    a = Kp_a * vr_e + vr_integral + Kd_a * (vr_e - prev_vr_e) / t_r
    a = np.clip(a, parameters[8], parameters[10])
    prev_vr_e = vr_e

    # Steering PID
    delta_e_integral = np.clip(delta_e_integral + Ki_delta * delta_e * t_d,
                               -max_integral_delta, max_integral_delta)
    raw_delta_cmd = Kp_delta * delta_e + delta_e_integral + Kd_delta * (delta_e - prev_delta_e) / t_d

    alpha = 0.72
    delta_cmd = alpha * raw_delta_cmd + (1 - alpha) * delta_cmd_prev
    delta_cmd_prev = delta_cmd

    delta_cmd = np.clip(delta_cmd, parameters[7], parameters[9])
    prev_delta_e = delta_e

    return np.array([delta_cmd, a])

def get_curvature(N: int, idx: int, racetrack: RaceTrack, points: list[int]):
    prev_pt  = racetrack.centerline[(idx + points[0]) % N]
    center   = racetrack.centerline[(idx + points[1]) % N]
    next_pt  = racetrack.centerline[(idx + points[2]) % N]

    h1 = np.arctan2(prev_pt[1] - center[1], prev_pt[0] - center[0])
    h2 = np.arctan2(center[1] - next_pt[1], center[0] - next_pt[0])
    angle_change = np.arctan2(np.sin(h2 - h1), np.cos(h2 - h1))

    path_dist = np.linalg.norm(next_pt - prev_pt)
    return abs(angle_change) / path_dist

def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    global finished, max_idx, last_lookahead, prev_delta_r
    sx, sy, _, v, heading = state

    N = len(racetrack.centerline)
    car_pos = np.array([sx, sy])

    distances = np.linalg.norm(racetrack.centerline - car_pos, axis=1)
    min_idx = np.argmin(distances)
    idx = max(min_idx, max_idx)
    min_dist = distances[min_idx]

    # Current curvature
    curvature = get_curvature(N, idx, racetrack, [0, 1, 2])

    # Lookahead curvatures
    lookahead_indices = [i for i in range(2, max(12, 6 + int(v // 4)), 2)]
    future_curvatures = [
        1 / np.log2(i + 2) * get_curvature(N, (idx + val) % N, racetrack, [-1, 0, 1])
        for i, val in enumerate(lookahead_indices)
    ]
    max_future_curv = max(future_curvatures)
    effective_curv = max(curvature, max_future_curv)

    # Extended preview for straights
    extended_preview = int(min(30, 12 + v * 0.4))
    preview_curvatures = [
        get_curvature(N, (idx + i) % N, racetrack, [-1, 0, 1])
        for i in range(1, extended_preview)
    ]
    avg_preview_curv = np.mean(preview_curvatures)
    max_preview_curv = max(preview_curvatures)

    is_on_straight = curvature < 0.003 and avg_preview_curv < 0.005
    curve_approaching = max_preview_curv > 0.015 and avg_preview_curv < 0.008

    curv_norm = np.clip(effective_curv / 0.02, 0, 3)
    
    # Calculate lookahead
    lookahead = int(3 + 6 * np.exp(-4 * curv_norm))
    if last_lookahead < lookahead:
        lookahead = max(int(np.ceil(last_lookahead + (lookahead - last_lookahead) / 4.0)),
                        last_lookahead + 1)

    # Calculate sharp turn
    severity = min(1.0, effective_curv / 0.02)
    sharp_turn = 0.75 + 3.5 * (severity ** 10)

    if effective_curv > 0.1:
        lookahead = 1
        sharp_turn = 5

    if idx + lookahead >= N or finished:
        finished = True
        idx, lookahead = 0, 0
        curvature = 10

    # Find target
    target = racetrack.centerline[(idx + lookahead) % N]
    tx, ty = target

    # Calculate velocity
    base_speed = parameters[5]
    v_r = base_speed / (1.0 + sharp_turn + 30 * np.sqrt(effective_curv))

    if is_on_straight and min_dist < 2.5:
        v_r = min(v_r * 1.15, base_speed)
    elif curve_approaching and v > 15:
        v_r *= 0.92

    v_r = np.clip(v_r, parameters[2], parameters[5])
    
    # Calculate delta
    phi_r = np.arctan2(ty - sy, tx - sx)
    e_phi = np.arctan2(np.sin(phi_r - heading), np.cos(phi_r - heading))

    steer_gain = 2.9
    if min_dist > 6:
        steer_gain = 1.0
    wheelbase = parameters[0]
    raw_delta_r = steer_gain * (wheelbase / max(v_r, 1.0)) * e_phi

    alpha = 0.5 + 0.40 * (1 - severity)
    delta_r = alpha * raw_delta_r + (1 - alpha) * prev_delta_r
    prev_delta_r = delta_r

    delta_r = np.clip(delta_r, parameters[1], parameters[4])
    max_idx = max(idx, max_idx)
    last_lookahead = lookahead
    
    return np.array([delta_r, v_r])

#