from MCBrainE25R5 import *
import scipy.io


def calculate_difference_distributions(countsdict):
    difference_distributions = {i: [] for i in range(4)}

    for (start, end), count in countsdict.items():
        diff = [abs(e - s) for s, e in zip(start, end)]
        for i in range(4):
            difference_distributions[i].extend([diff[i]] * count)

    return difference_distributions

if __name__ == "__main__":
    loaded_count_array = np.load('E25R5DT001I0625JochCounts.npy', allow_pickle=True)
    loaded_counts_dict = defaultdict(int, dict(loaded_count_array))
    print('loadedcounts')
    difference_distributions = calculate_difference_distributions(loaded_counts_dict)

    for i in range(4):
        print(f'Difference distribution for index {i}:', difference_distributions[i])
