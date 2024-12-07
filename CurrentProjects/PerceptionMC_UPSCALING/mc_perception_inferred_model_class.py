from mc_perception_frwd_cl import simulation_nopij

""" Class for brain model, holds trajectories and provides methods for trajectory generation, analysis, and plotting. """

class brain_model:
    def __init__(self, params, maxtop, maxbot):
        self.params = params
        self.maxtop = maxtop
        self.maxbot = maxbot
        self.trajectories = []

    # # # Trajectory Generation # # #
#############################################################################
    def generate_trajectory(self, initial_state, tmax):

        inferred_trajectory = simulation_nopij(initial_state, tmax, self.params)
        self.trajectories.append(inferred_trajectory)

        return inferred_trajectory

    def save_trajectory(self, file_path):



        pass

    def load_trajectory(self, file_path):
        """
        Load trajectories from a file.
        """
        pass

    # # # Trajectory Analysis # # #
#############################################################################
    def analyze_trajectories(self):
        """
        Analyze the loaded or generated trajectories.
        """
        pass


    # # # Plotting # # #
#############################################################################
    def plot_trajectory(self, trajectory):
        """
        Plot a given trajectory.
        """
        pass

    def save_plot(self, plot, file_path):
        """
        Save the plot to a file.
        """
        pass