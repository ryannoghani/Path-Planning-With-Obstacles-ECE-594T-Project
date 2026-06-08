import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

N_EXAMPLES = 5                          # Number of random examples to generate
OUTPUT_FOLDER = "Experiment_Results"    # Folder where images will be saved
radius = .1                             # Square obstacle half-width
num_obstacles = 5                       # Number of obstacles per example
tol = 1e-3                              # Convergence threshold
K = 21                                  # How many points in our trajectory
x0 = [0, 0]                             # Starting point
xK = [1, 1]                             # Ending point

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def Run_Relaxation_problem(radius, centers, x0, xK):
    X = [cp.Variable((5, 5), PSD=True) for k in range(K-1)]

    # Objective function
    obj = 0
    P = np.zeros((5, 5))    
    P[1:, 1:] = np.eye(4)
    P[1:3, 3:] = - np.eye(2)
    P[3:, 1:3] = - np.eye(2)
    for Xk in X:
        obj += cp.trace(P @ Xk)

    # Initial and final conditions
    constraints = [
        X[0][0, 1:3] == x0,
        X[0][1:3, 1:3] == cp.outer(x0, x0),
        X[0][1:3, 3:] == cp.outer(x0, X[0][0, 3:]),
        X[-1][0, 3:] == xK,
        X[-1][3:, 3:] == cp.outer(xK, xK),
        X[-1][1:3, 3:] == cp.outer(X[-1][0, 1:3], xK)
    ]

    # Coherence constraints
    for Xk in X:
        constraints.append(Xk[0, 0] == 1)
    for Xk, Xkp in zip(X[:-1], X[1:]):
        constraints.append(Xk[0, 3:] == Xkp[0, 1:3])
        constraints.append(Xk[3:, 3:] == Xkp[1:3, 1:3])

    # Obstacle avoidance constraints (Stage 1: Base Inner Radius Match)
    for Xk in X:
        for center in centers:
            P_obs = np.zeros((5, 5))
            P_obs[0, 0] = radius ** 2 - center.dot(center)
            P_obs[0, 1:3] = center
            P_obs[1:3, 0] = center
            P_obs[1:3, 1:3] = - np.eye(2)
            constraints.append(cp.trace(P_obs @ Xk) <= 0)

    # Solve problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver='MOSEK')
    
    points = np.array([Xk[0, 1:3].value for Xk in X] + [X[-1][0, 3:].value])
    return points


def objective_function(points):
    return cp.sum_squares(points[1:] - points[:-1])


# =============================================================================
# EXACT PIECEWISE-LINEAR ENVELOPE METHOD FOR MIN_t ||a + bt||_inf
# =============================================================================
def find_t_star_infinity_norm(p_curr, p_next, c):
    """
    Finds the exact t in [0, 1] that minimizes ||(1-t)*p_curr + t*p_next - c||_inf
    Using the analytical piecewise-linear candidate selection.
    """
    a = p_curr - c
    b = p_next - p_curr
    
    a1, a2 = a[0], a[1]
    b1, b2 = b[0], b[1]
    
    # Base endpoint bounds
    candidates = [0.0, 1.0]
    
    # 1. Slope changes where individual coordinate components cross zero
    if np.abs(b1) > 1e-9:
        candidates.append(-a1 / b1)
    if np.abs(b2) > 1e-9:
        candidates.append(-a2 / b2)
        
    # 2. Breakpoints where the identity of the maximizing coordinate changes
    # Case A: a1 + b1*t = a2 + b2*t
    if np.abs(b1 - b2) > 1e-9:
        candidates.append((a2 - a1) / (b1 - b2))
        
    # Case B: a1 + b1*t = -(a2 + b2*t)
    if np.abs(b1 + b2) > 1e-9:
        candidates.append((-a1 - a2) / (b1 + b2))
        
    # Evaluate valid candidates inside the active segment domain
    best_t = 0.0
    min_dist = float('inf')
    
    for t in candidates:
        if 0.0 <= t <= 1.0:
            p_test = p_curr + t * (p_next - p_curr)
            dist_test = np.linalg.norm(p_test - c, ord=np.inf)
            if dist_test < min_dist:
                min_dist = dist_test
                best_t = t
                
    p_worst = p_curr + best_t * (p_next - p_curr)
    return best_t, p_worst


# =============================================================================
# CHOSEN FACE COMPONENT LINEARIZATION
# =============================================================================
def linearized_square_sip_constraint(p_curr, p_next, p_curr_new, p_next_new, c, radius):
    """
    Finds the exact worst-case intersection parameter t_star, evaluates which
    flat face wall is breached, and returns linear constraints bounding the new variables.
    """
    t_star, p_worst = find_t_star_infinity_norm(p_curr, p_next, c)
    
    # Establish the optimization variable position map matching the scalar t_star
    p_worst_new = (1 - t_star) * p_curr_new + t_star * p_next_new
    diff = p_worst - c
    
    # Apply standard linear bounding plane inequalities matching the dominant face axis
    if np.abs(diff[0]) >= np.abs(diff[1]):
        if diff[0] >= 0:
            return [p_curr_new[0] >= c[0] + radius, p_next_new[0] >= c[0] + radius]
        else:
            return [p_curr_new[0] <= c[0] - radius, p_next_new[0] <= c[0] - radius]
    else:
        if diff[1] >= 0:
            return [p_curr_new[1] >= c[1] + radius, p_next_new[1] >= c[1] + radius]
        else:
            return [p_curr_new[1] <= c[1] - radius, p_next_new[1] <= c[1] - radius]


# =============================================================================
# MAIN AUTOMATION EXECUTION LOOP
# =============================================================================
for run in range(1, N_EXAMPLES + 1):
    print(f"Generating example {run}/{N_EXAMPLES}...")

    # Center placement bounds safely padded from workspace margins
    centers = np.random.uniform(0.15 + radius, 0.85 - radius, size=(num_obstacles, 2))

    # Stage 1: Call global relaxation model
    initial_points = Run_Relaxation_problem(radius, centers, x0, xK)

    solutions = [initial_points] 
    values = [objective_function(initial_points).value] 
    failed = False 

    # Stage 2: Local SIP refinement loop
    max_iters = 35
    for iteration in range(max_iters):
        points = solutions[-1] 
        new_points = cp.Variable(points.shape)  
        constraints = [
            new_points[0] == points[0], 
            new_points[-1] == points[-1] 
        ]

        # Scan every continuous segment against every square obstacle
        for k in range(len(points) - 1):
            p_curr = points[k]
            p_next = points[k+1]
            p_curr_new = new_points[k]
            p_next_new = new_points[k+1]
            
            for c in centers:
                # Find the analytical infinity-norm distance to this obstacle
                _, p_worst = find_t_star_infinity_norm(p_curr, p_next, c)
                segment_dist = np.linalg.norm(p_worst - c, ord=np.inf)
                
                # Only activate the face walls if the segment penetrates or clips the obstacle bounding box
                if segment_dist < radius + 1e-3:
                    face_constraints = linearized_square_sip_constraint(p_curr, p_next, p_curr_new, p_next_new, c, radius)
                    constraints.extend(face_constraints)
                
        obj = objective_function(new_points)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            prob.solve(solver='MOSEK')
            if prob.status not in ["optimal", "optimal_inaccurate"] or new_points.value is None:
                failed = True
                break
        except Exception:
            failed = True
            break
            
        solutions.append(new_points.value)
        values.append(prob.value)

        # Terminate when the decrease in objective function flattens out
        if abs(values[-2] - values[-1]) / values[-1] < tol:
            break
        
    if failed:
        print(f"  Skipping run {run}: Local solver failure or infeasible geometric layout.")
        continue

    # Render Visual Plot Map
    plt.figure()
    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    # Draw square obstacles using Rectangle patches
    for c in centers:   
        bottom_left = [c[0] - radius, c[1] - radius]
        side_length = 2 * radius
        patch = Rectangle(bottom_left, side_length, side_length, facecolor='lightcoral', edgecolor='black', zorder=1)
        plt.gca().add_patch(patch)
        
    # Render development history tracking line paths
    for idx, (pts, value) in enumerate(zip(solutions, values)):
        rounded_value = np.round(value, 5)
        alpha_val = 1.0 if idx == len(solutions)-1 else 0.2
        lw_val = 2 if idx == len(solutions)-1 else 1
        lbl = f"Final Iteration (Obj = {rounded_value})" if idx == len(solutions)-1 else f"Iter {idx} (Obj = {rounded_value})"
        plt.plot(*pts.T, marker='o', alpha=alpha_val, linewidth=lw_val, label=lbl, zorder=2)
        
    plt.title(f"SIP Square Trajectory Optimization Example {run}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    file_path = os.path.join(OUTPUT_FOLDER, f"example_{run:03d}.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=150)
    plt.close() 

print(f"\nDone! All valid square trajectory maps have been saved to the '{OUTPUT_FOLDER}' directory.")