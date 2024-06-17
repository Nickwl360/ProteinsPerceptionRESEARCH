from MCBrainE25R5 import *
import scipy.io


def calculateDifferenceDistributions(countsDict):
    # Initialize dictionaries to store the distribution of differences for each index
    differenceDistributions = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int)}

    for (start, end), count in countsDict.items():
        # Calculate differences for each index
        diff1 = abs(start[0] - end[0])
        diff2 = abs(start[1] - end[1])
        diff3 = abs(start[2] - end[2])
        diff4 = abs(start[3] - end[3])

        # Update the distributions for each index
        differenceDistributions[1][diff1] += count
        differenceDistributions[2][diff2] += count
        differenceDistributions[3][diff3] += count
        differenceDistributions[4][diff4] += count

    return differenceDistributions

if __name__ == "__main__":
    loaded_count_array = np.load('E25R5DT001I0625JochCounts.npy', allow_pickle=True)
    loaded_counts_dict = defaultdict(int, dict(loaded_count_array))
    print('loaded')
    differenceDistributions = calculateDifferenceDistributions(loaded_counts_dict)

    # Print the distributions for each index
    for index, distribution in differenceDistributions.items():
        print(f"Index {index} difference distribution:")
        for difference, count in sorted(distribution.items()):
            print(f"Difference {difference}: {count} times")

