def winning_strategy(strat_input):
    team = strat_input.team
    my_coords = strat_input.player_coords[team].copy()
    opp_coords = strat_input.player_coords[1 - team].copy()
    ball = strat_input.ball
    state = strat_input.prev_state or {'tick': 0}
    state['tick'] += 1

    def mirror(c): return -c if team == 1 else c
    m_my = mirror(my_coords)
    m_opp = mirror(opp_coords)
    m_ball = mirror(ball.coords)

    new_m_coords = np.zeros((5, 2))

    # --- 1. The High-Resolution Move Engine ---
    # 360 candidates ensure we always find a mathematically legal move.
    radii = np.array([1.05, 3.0, 5.05, 7.5, 9.95]) 
    angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
    R, A = np.meshgrid(radii, angles)
    offsets = np.column_stack([R.ravel() * np.cos(A.ravel()), R.ravel() * np.sin(A.ravel())])

    def get_best_safe_move(current, target, min_spacing):
        candidates = current + offsets
        valid_x = (candidates[:, 0] >= -49.9) & (candidates[:, 0] <= 49.9)
        valid_y = (candidates[:, 1] >= -24.9) & (candidates[:, 1] <= 24.9)
        valid_mask = valid_x & valid_y

        if min_spacing > 0:
            dists_to_ball = np.linalg.norm(candidates - m_ball, axis=1)
            valid_mask &= (dists_to_ball >= min_spacing)

        if not np.any(valid_mask):
            vec = np.array([0.0, 0.0]) - current
            return current + (vec / (np.linalg.norm(vec) + 1e-6)) * 1.05

        valid_candidates = candidates[valid_mask]
        dists_to_target = np.linalg.norm(valid_candidates - target, axis=1)
        return valid_candidates[np.argmin(dists_to_target)]

    # --- Fast Segment Intersection Helper ---
    def pt_to_seg(p, a, b):
        ab = b - a
        ap = p - a
        t = np.clip(np.sum(ap * ab, axis=1) / (np.sum(ab * ab) + 1e-8), 0, 1)
        closest = a + t[:, None] * ab
        return np.linalg.norm(p - closest, axis=1)

    # --- 2. Mode: The Trajectory Spear (Loose Ball) ---
    if ball.team is None:
        # We shoot our players directly THROUGH the ball to win the trajectory intersection.
        for i in range(5):
            vec = m_ball - m_my[i]
            target = m_my[i] + (vec / (np.linalg.norm(vec) + 1e-6)) * 15.0 
            new_m_coords[i] = get_best_safe_move(m_my[i], target, 0.0)

    # --- 3. Mode: The Dynamic Eclipse Arc (Defense) ---
    elif ball.team == 1 - team:
        # Calculate exact angles to the top and bottom of our goal line.
        ang_top = np.arctan2(24.9 - m_ball[1], -50.0 - m_ball[0])
        ang_bot = np.arctan2(-24.9 - m_ball[1], -50.0 - m_ball[0])
        
        # Unwrap angles to properly wrap around the back of the ball
        if ang_bot < 0 and ang_top > 0: 
            ang_bot += 2 * np.pi
            
        arc_angles = np.linspace(ang_top, ang_bot, 5)
        targets = np.array([m_ball + 4.05 * np.array([np.cos(a), np.sin(a)]) for a in arc_angles])
        
        # Sort targets and players by Y to avoid players crossing paths
        targets = targets[np.argsort(targets[:, 1])]
        y_sorted_indices = np.argsort(m_my[:, 1])
        
        for tgt_idx, player_idx in enumerate(y_sorted_indices):
            new_m_coords[player_idx] = get_best_safe_move(m_my[player_idx], targets[tgt_idx], 4.05)

    # --- 4. Mode: The Omni-Scanner (Offense) ---
    else:
        idx = ball.player
        scan_angles = np.linspace(0, 2*np.pi, 72)
        shots = m_ball + 19.5 * np.column_stack([np.cos(scan_angles), np.sin(scan_angles)])
        
        shots[:, 0] = np.clip(shots[:, 0], -49.9, 49.9)
        shots[:, 1] = np.clip(shots[:, 1], -24.9, 24.9)
        
        best_shot = None
        best_score = -np.inf
        
        # Scan every lane for the optimal, completely unblockable pass 
        for shot in shots:
            dists = pt_to_seg(m_opp, m_ball, shot)
            min_d = np.min(dists)
            
            # 2.0m clearance defeats the 1m REACH and the opponent's movement
            if min_d > 2.0: 
                score = shot[0] # Prioritize forward (+X) distance
                if score > best_score:
                    best_score = score
                    best_shot = shot
                    
        # Panic Fallback: Find the widest gap if all forward lanes are heavily guarded
        if best_shot is None:
            max_gap = -1
            for shot in shots:
                dists = pt_to_seg(m_opp, m_ball, shot)
                min_d = np.min(dists)
                if min_d > max_gap:
                    max_gap = min_d
                    best_shot = shot

        # Execute the shot within 3-20m constraints
        shot_vec = best_shot - m_ball
        shot_dist = np.linalg.norm(shot_vec)
        new_m_coords[idx] = m_ball + (shot_vec / (shot_dist + 1e-6)) * np.clip(shot_dist, 3.1, 19.9)
        
        # Teleport Swarm: All other players sprint to catch the perfect pass
        for i in range(5):
            if i != idx:
                new_m_coords[i] = get_best_safe_move(m_my[i], best_shot, 5.05)

    return StrategyOutput(mirror(new_m_coords), state)