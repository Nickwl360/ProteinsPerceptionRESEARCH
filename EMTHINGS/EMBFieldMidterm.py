import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##biggerparticle options
Z=1

def Bfield(position):
    M = np.array([0, 0, -8e22])
    x, y, z = position
    r = np.sqrt(x**2 + y**2 + z**2)
    u0 = 12.57e-7
    if r == 0:
        return np.array([0, 0, 0])
    pos_unit = position / r  # Unit vector of position
    B = (u0 / (4 * np.pi)) * (1 / r**3) * (3 * np.dot(M, pos_unit) * pos_unit - M)
    return B

class Proton:
    def __init__(self, position, velocity, charge=1.602e-19, mass=1.672e-27):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.charge = charge
        self.mass = mass*Z
        self.initial_speed = np.linalg.norm(velocity)

    def update_position(self, dt):
        ###MAybe RK 4th
        self.position += self.velocity * dt

    def update_velocity(self, magnetic_field, dt):
        force = self.charge * np.cross(self.velocity, magnetic_field)  #F = Q VXB
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        current_speed = np.linalg.norm(self.velocity)
        if current_speed != 0:
            self.velocity *= self.initial_speed / current_speed

def simulate_proton_trajectory(proton, total_time, dt):
    positions = []
    N = int(total_time / dt)
    for step in range(N):
        if step % 1000 == 0:  # Print every 1000 steps
            print(f"Step {step}/{N}")
            print(f"Position: {proton.position}")
            print(f"Velocity: {proton.velocity}")
            print(f"Speed: {np.linalg.norm(proton.velocity)}")

        positions.append(proton.position.copy())
        B = Bfield(proton.position)
        proton.update_velocity(B, dt)
        proton.update_position(dt)
    return positions

def plot_trajectory(trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-',linewidth=.1)

    radius_earth = 6371e3  # Radius of Earth in meters
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius_earth * np.outer(np.cos(u), np.sin(v))
    y = radius_earth * np.outer(np.sin(u), np.sin(v))
    z = radius_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.3, rstride=4, cstride=4)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Proton Trajectory in Earth\'s Magnetic Field')

    max_val = 1.5 * radius_earth
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)

    plt.show()

def plot_magnetic_field():

    grid_size = 10
    x = np.linspace(-10e6, 10e6, grid_size)
    y = np.linspace(-10e6, 10e6, grid_size)
    z = np.linspace(-10e6, 10e6, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    #bfieldstorage
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    W = np.zeros_like(Z)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                position = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                B = Bfield(position)
                U[i, j, k] = B[0]
                V[i, j, k] = B[1]
                W[i, j, k] = B[2]

    # Plot magnetic field lines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=1e6, normalize=True, alpha=0.5)

    # Set labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Magnetic Field around Earth')

    # Adjust axis limits
    max_val = 10e6
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)

    plt.show()

if __name__ == "__main__":
    plot_magnetic_field()

    #######INITIALCONDITIONS############
    # initial_position = [7371e3, 0, 8000e3]
    # initial_velocity = [0,0 , -6e7]
    # proton = Proton(initial_position, initial_velocity)
    #
    # total_time = 2.8  # Total time in seconds
    # dt = 1e-3  # Time step in seconds
    #
    # trajectory = simulate_proton_trajectory(proton, total_time, dt)
    #
    # plot_trajectory(trajectory)
