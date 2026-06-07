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
K = 21                                  # How many points in our trajectoryd

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


initial_points = np.vstack([np.linspace([0, 0], [1.0, 0], 10),
                            np.linspace([1.0, 0], [1.0, 1.0], 10)
                            ])



def objective_function(points):
    return cp.sum_squares(points[1:] - points[:-1]) #Sum of squared distances

# Take the 2nd term in each constraint and do a 2nd order approximation of it. This is for the relaxation of nonconvex constraints
def linearized_constraint(p, p_new, c):
    diff = p - c
    dist = np.linalg.norm(diff, ord = np.inf) # BIGGEST CHANGE: Set the norm to infinity norm
    offset = radius - dist

# Next 3 lines show general formula for gradient of p norm
    # Because the infinity norm is not differentiable and has a piecewise graident, 
    # I figure out which absolute value component of diff is largest
    if np.abs(diff[0]) > np.abs(diff[1]):
        gradient = np.array([np.sign(diff[0]), 0.0])
    else:   # Implicitly this handles the = case, so this is computing a subgradient
        gradient = np.array([0.0, np.sign(diff[1])])

    return offset - gradient @ (p_new - p) <= 0




for run in range(1, N_EXAMPLES + 1):
    print(f"Generating example {run}/{N_EXAMPLES}...")

    # Keep between 0.1 and 0.9 because radius is 0.1 so that we dont exceed borders
    centers = np.random.uniform(0.1, 0.9, size=(num_obstacles, 2))

    solutions = [initial_points] # stores trajectories at all iterations for plotting
    values = [objective_function(initial_points).value] # stores objective values at all iterations for showing evolution
    
    # Flag to check if the solver fails or becomes infeasible due to bad random geometry
    failed = False 

    
    while True:
        
        points = solutions[-1] # previous trajectory
        new_points = cp.Variable(points.shape)  #K different positions
        constraints = [
            new_points[0] == points[0], # initial point is fixed
            new_points[-1] == points[-1] # kth final point is fixed
        ]

        for p, p_new in zip(points, new_points):
            for c in centers:
                # linearized obstacle avoidance
                constraints.append(linearized_constraint(p, p_new, c))
                
        obj = objective_function(new_points)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            prob.solve()
            # This will check if CVXPY fails to find a solution (Infeasible/Unbounded).
            # Im doing try catch so I can exit the loop and keep the results CVXPY was able to solve.
            if prob.status not in ["optimal", "optimal_inaccurate"] or new_points.value is None:
                failed = True
                break
        except Exception:
            failed = True
            break
        # Convergence check.
        if (values[-1] - prob.value) / prob.value < tol:
            break
        solutions.append(new_points.value)
        values.append(prob.value)
        
    if failed:
        print(f"  Skipping run {run}: Obstacles generated an impossible/infeasible setup.")
        continue

    # Plot result.
    plt.figure()
    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    for c in centers:   #To draw the squares, we use the bottom left corner and side length of each square
        bottom_left = [c[0] - radius, c[1] - radius]
        side_length = 2 * radius
        patch = Rectangle(bottom_left, side_length, side_length, facecolor='lightcoral', edgecolor='black')
        plt.gca().add_patch(patch)
        
    # Draw trajectory evolution
    #Credit to AI for this part (Google Gemini)
    for idx, (points, value) in enumerate(zip(solutions, values)):
        rounded_value = np.round(value, 5)
        # Plot the final iteration with a thicker line
        alpha_val = 1.0 if idx == len(solutions)-1 else 0.3
        lw_val = 2 if idx == len(solutions)-1 else 1
        plt.plot(*points.T, marker='o', label=f"Objective value = {rounded_value}")
        
    plt.title(f"Random Trajectory Optimization Example {run}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Push legend outside plot
    
    # Save the file into our output folder
    file_path = os.path.join(OUTPUT_FOLDER, f"example_{run:03d}.png")
    plt.savefig(file_path, bbox_inches='tight', dpi=150)
    plt.close() # Close figure to free up system memory

print(f"\nDone! All valid trajectory maps have been saved to the '{OUTPUT_FOLDER}' directory.")
