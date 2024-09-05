import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Draw the bridge deck
bridge_length = 10
deck_y = 1
ax.plot([0, bridge_length], [deck_y, deck_y], color='black', linewidth=4)

# Draw the bridge piers (vertical supports)
pier_positions = [2, 5, 8]
for x in pier_positions:
    ax.plot([x, x], [0, deck_y], color='brown', linewidth=6)

# Draw the suspension cables (parabolic shape)
cable_x = np.linspace(0, bridge_length, 1000)
cable_y = -0.02 * (cable_x - bridge_length/2)**2 + 3
ax.plot(cable_x, cable_y, color='blue', linewidth=2)

# Draw the vertical hangers connecting the deck to the cables
num_hangers = 20
hanger_positions = np.linspace(0, bridge_length, num_hangers)
hanger_heights = -0.02 * (hanger_positions - bridge_length/2)**2 + 3
for i in range(num_hangers):
    ax.plot([hanger_positions[i], hanger_positions[i]], [deck_y, hanger_heights[i]], color='gray', linewidth=1.5)

# Set the limits and labels
ax.set_xlim(-1, bridge_length + 1)
ax.set_ylim(0, 4)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Length of the Bridge')
ax.set_ylabel('Height')

# Show the plot
plt.show()
