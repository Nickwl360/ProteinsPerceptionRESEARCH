import numpy as np
import matplotlib.pyplot as plt

def Bfield(position):
    M = np.array([0, 0, -8e22])
    x, y, z = position
    r = np.sqrt(x**2 + y**2 + z**2)
    u0 = 12.57e-7
    if r == 0:
        return np.array([0, 0, 0])
    pos_unit = position / r
    B = (u0 / (4 * np.pi)) * (1 / r**3) * (3 * np.dot(M, pos_unit) * pos_unit - M)
    return B

def getStreamLine(start, steps, dt):
    positions = [start]
    currentR = start
    for i in range(steps):
        B = Bfield(currentR)
        if np.linalg.norm(B) == 0:
            break
        nextR = currentR + B / np.linalg.norm(B) * dt
        positions.append(nextR)
        currentR = nextR
    return np.array(positions)

def PlotBfieldEarth():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    initialpoints = [
        [6e6, 0, 0],
        [-6e6, 0, 0],
        [0, 6e6, 0],
        [0, -6e6, 0],
        [0, 0, 6e6],
        [0, 0, -6e6],
        [6e6, 6e6, 0],
        [-6e6, -6e6, 0],
        [6e6, -6e6, 0],
        [-6e6, 6e6, 0],
        [6e6, 0, 6e6],
        [-6e6, 0, -6e6],
        [0, 6e6, 6e6],
        [0, -6e6, -6e6]
    ]
    steps = 2000
    step_size = 1e5
    for start_pos in initialpoints:
        streamline = getStreamLine(np.array(start_pos), steps, step_size)
        ax.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2], 'b-',linewidth=.5)

    # Plot the Earth as a sphere #GOT THIS FROM CHAT GPT TO PLOT A SPHERE
    radius_earth = 6371e3  # Radius of Earth in meters
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius_earth * np.outer(np.cos(u), np.sin(v))
    y = radius_earth * np.outer(np.sin(u), np.sin(v))
    z = radius_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.3, rstride=4, cstride=4)
    # Set labels and title
    ax.set_xlabel('X Position (1e7m)')
    ax.set_ylabel('Y Position (1e7m)')
    ax.set_zlabel('Z Position (1e7m)')
    ax.set_title('Magnetic Field Streamlines around Earth')
    max_val = 10e6
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)

    plt.show()

if __name__ == "__main__":
    PlotBfieldEarth()
