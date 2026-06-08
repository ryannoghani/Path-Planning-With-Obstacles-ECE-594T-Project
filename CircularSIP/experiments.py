import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

N_EXAMPLES = 5                          # Number of random examples to generate
OUTPUT_FOLDER = "Experiment_Results"    # Folder where images will be saved
radius = .1                             # Circle obstacle radius
num_obstacles = 5                       # Number of obstacles per example
tol = 1e-3                              # Convergence threshold
K = 21                                  # How many points in our trajectory
x0 = [0, 0]                             # Starting point
xK = [1, 1]                             # Ending point

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def Run_Relaxation_problem(radius, centers, x0, xK):
    # X(k) = [1, x1(k) x2(k) x1(k+1) x2(k+1)] * [1, x1(k) x2(k) x1(k+1) x2(k+1)]^T
    X = [cp.Variable((5, 5), PSD=True) for k in range(K-1)]

    # Objective function
    obj = 0
    #     | 0   0   0   0   0 |
    #     | 0   1   0  -1   0 |
    # P = | 0   0   1   0  -1 |
    #     | 0  -1   0   1   0 |
    #     | 0   0  -1   0   1 |
    P = np.zeros((5, 5))    
    P[1:, 1:] = np.eye(4)
    P[1:3, 3:] = - np.eye(2)
    P[3:, 1:3] = - np.eye(2)
    for Xk in X:
        obj += cp.trace(P @ Xk)

    # Initial and final conditions
    constraints = [
        # [x1(0), x2(0)] == [0, 0]
        X[0][0, 1:3] == x0,

        # Make sure X contains x0x0^T
        X[0][1:3, 1:3] == cp.outer(x0, x0),

        # Transition Constraints to X[1]
        X[0][1:3, 3:] == cp.outer(x0, X[0][0, 3:]),

        # [x1(K), x2(K)] = [1, 1]
        X[-1][0, 3:] == xK,

        # Make sure X contains xKxK^T
        X[-1][3:, 3:] == cp.outer(xK, xK),

        # Transition Constraints from X[K-1]
        X[-1][1:3, 3:] == cp.outer(X[-1][0, 1:3], xK)
    ]

    # Coherence constraints
    for Xk in X:
        constraints.append(Xk[0, 0] == 1)
    for Xk, Xkp in zip(X[:-1], X[1:]):
        constraints.append(Xk[0, 3:] == Xkp[0, 1:3])
        constraints.append(Xk[3:, 3:] == Xkp[1:3, 1:3])

    # Obstacle avoidance constraints
    for Xk in X:
        for center in centers:
            P = np.zeros((5, 5))
            P[0, 0] = radius ** 2 - center.dot(center)
            P[0, 1:3] = center
            P[1:3, 0] = center
            P[1:3, 1:3] = - np.eye(2)
            constraints.append(cp.trace(P @ Xk) <= 0)

    # Solve problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver='MOSEK')
    
    # Extract optimal solution
    points = np.array([Xk[0, 1:3].value for Xk in X] + [X[-1][0, 3:].value])
    return points


def objective_function(points):
    return cp.sum_squares(points[1:] - points[:-1])


# Take the 2nd term in each constraint and do a 1st order approximation of it. This is for the relaxation of nonconvex constraints
def linearized_sip_constraint(x_k, x_k1, x_new_k, x_new_k1, c, radius):
    # We are going to compute the minimizer t_star. I talk about how to solve t_star in the report.
    denominator = (x_k1 - x_k) @ (x_k1 - x_k)
    if denominator < 1e-8:     # This is to prevent division by 0
        t_star = 0.0    #t_star is extremely large negative number and needs to be projected back to 0.
    else:
        t_star = -((x_k1 - x_k) @ (x_k - c)) / denominator
        t_star = np.clip(t_star, 0.0, 1.0) # Project back to either 0 or 1 if out of range

    # Plug in t_star to find the closest our trajectory between k and k+1 gets to the center of circle.
    closest_to_center = (1 - t_star) * x_k + t_star * x_k1
    
    # Calculate the closest distance we are from the circle
    min_distance = np.linalg.norm(closest_to_center - c)
    if min_distance < 1e-6: # Prevents dividing by 0 later on
        min_distance = 1e-6
        
    offset = radius - min_distance  # This is sort of our constraint. Offset must be <= 0
    gradient = - (closest_to_center - c) / min_distance
    
    # Below is just a fancy way of writing the linearized restriction. 
    # I just factored out (1 - t_star)
    return offset + (1 - t_star) * gradient @ (x_new_k - x_k) + t_star * gradient @ (x_new_k1 - x_k1) <= 0


# Running experiments
for run in range(1, N_EXAMPLES + 1):
    print(f"Generating example {run}/{N_EXAMPLES}...")

    centers = np.random.uniform(0.1, 0.9, size=(num_obstacles, 2))

    # we initialize our trajectory with the solution to the relaxation
    # this helps us get a global solution
    initial_points = Run_Relaxation_problem(radius, centers, x0, xK)
    solutions = [initial_points] # stores trajectories at all iterations for plotting
    values = [objective_function(initial_points).value] # stores objective values at all iterations for showing evolution
    failed = False 

    while True:
        points = solutions[-1] # previous trajectory
        new_points = cp.Variable(points.shape) # points that make sure the lines connecting them dont go thru obstacle
        constraints = [
            new_points[0] == points[0],  # initial point is fixed
            new_points[-1] == points[-1] # kth final point is fixed
        ]

        # Process constraints segment-by-segment to capture continuous geometry
        for k in range(len(points) - 1):
            x_k = points[k]
            x_k1 = points[k+1]
            x_new_k = new_points[k]
            x_new_k1 = new_points[k+1]
            
            for c in centers:
                # linearized obstacle avoidance
                constraints.append(
                    linearized_sip_constraint(x_k, x_k1, x_new_k, x_new_k1, c, radius)
                )
                
        obj = objective_function(new_points)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            prob.solve()
            if prob.status not in ["optimal", "optimal_inaccurate"] or new_points.value is None:
                failed = True
                break
        except Exception:
            failed = True
            break
            
        # Convergence check.
        if abs(values[-1] - prob.value) / prob.value < tol:
            break
        solutions.append(new_points.value)
        values.append(prob.value)
        
    if failed:
        print(f"  Skipping run {run}: Local solver failure or infeasible geometric layout.")
        continue

    # Plot result.
    plt.figure()
    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    for c in centers:   
        patch = Circle(c, radius, facecolor='lightcoral', edgecolor='black', zorder=1)
        plt.gca().add_patch(patch)
        
    for idx, (pts, value) in enumerate(zip(solutions, values)):
        rounded_value = np.round(value, 5)
        alpha_val = 1.0 if idx == len(solutions)-1 else 0.3
        lw_val = 2 if idx == len(solutions)-1 else 1
        lbl = f"Final Iteration (Obj = {rounded_value})" if idx == len(solutions)-1 else f"Iter {idx} (Obj = {rounded_value})"
        plt.plot(*pts.T, marker='o', alpha=alpha_val, linewidth=lw_val, label=lbl, zorder=2)
        
    plt.title(f"SIP Trajectory Optimization Example {run}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    file_path = os.path.join(OUTPUT_FOLDER, f"example_{run:03d}.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=150)
    plt.close() 

print(f"\nDone! All valid trajectory maps have been saved to the '{OUTPUT_FOLDER}' directory.")