import math
from collections import Counter


def Knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []
    # for each example in data
    for index, example in enumerate(data):
        # Calculate the distance between the query example and the current
        distance = distance_fn(example[:-1], query)
        # Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append(distance, index)

# Sort the ordered collection of distances and indices from
# smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(
        neighbor_distances_and_indices)


def euclidean_distance(point1, point2):
    # calculate the distance between the query example and the current

    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)


def mean(labels):
    return sum(labels)/len(labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]
