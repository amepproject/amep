import amep  # Custom plotting library (not a standard Python library)
import numpy as np
import matplotlib.pyplot as plt

def draw_arrows(fig, start_points, displacements, color="blue", alpha=0.8, width=0.005, head_width=0.02, head_length=0.03):
    """
    Draws arrows on a given figure at specified start points with defined displacements.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure on which the arrows will be drawn.
    start_points : list of tuples
        A list of (x, y) coordinates where arrows should start.
    displacements : list of tuples
        A list of (dx, dy) displacement vectors corresponding to each start point.
    color : str, optional
        The color of the arrows (default is "blue").
    alpha : float, optional
        The transparency level of the arrows (default is 0.8).
    width : float, optional
        The width of the arrow body (default is 0.005).
    head_width : float, optional
        The width of the arrowhead (default is 0.02).
    head_length : float, optional
        The length of the arrowhead (default is 0.03).
    """
    for (_x, _y), (dx, dy) in zip(start_points, displacements):
        print(f"Arrow from ({_x}, {_y}) with displacement ({dx}, {dy})")  # Debugging output

        # Draw an arrow on the figure
        amep.plot.draw_arrow(
            fig, _x, _y, dx, dy, 
            color=color, alpha=alpha, width=width, 
            head_width=head_width, head_length=head_length
        )

# Generate x values from 0 to 2π
x = np.linspace(0, 2 * np.pi, 100)

# Compute sine values directly
y = np.sin(x)

# Define arrow starting points and displacements
start_points = [(0.3, 0.6)]  # List of (x, y) start positions
displacements = [(0.25, 0.25)]  # Corresponding (dx, dy) displacement values

# Create a figure using the `amep.plot.new` function
fig, axs = amep.plot.new(figsize=(3, 3))

# Plot the sine function
axs.plot(x, y, color="black", linewidth=2)

# Call the function to draw arrows
draw_arrows(fig, start_points, displacements)