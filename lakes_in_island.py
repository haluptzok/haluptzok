# Welcome to Meta!

# This is just a simple shared plaintext pad, with no execution capabilities.

# When you know what language you would like to use for your interview,
# simply choose it from the dropdown in the left bar.

# Enjoy your interview!

# Problem: K-means Clustering

# Given a dataset of points - m samples of n dimensional vectors - k cluster centers that represent the data

# One simple method is - just merge the 2 closest vectors - repeat - until just k left.

# Merging the vectors - minimize the approximation error introduced - so each cluster needs a weight

import numpy as np
import time

def KMeansCluster(pData, kClusters):  # pData is m X n matrix of samples list of lists, k is the desired number to return
    m = len(pData)    # number of samples
    n = len(pData[0]) # dimension of each vector
    weight = [1.0] * m  # each point is initially weight 1
    print("Start", kClusters, "clusters")

    # compute for each point the nearest neighbors
    # find the 2 closest and merge them into a new point
    # the distance should include the approximation introduced by factoring the weight
    # merge points - keep deleting from the end of the array

    while m > kClusters:
        # print("while start m", m, "k", kClusters)
        # find the closest 2 points to merge
        min_dist = 1000000000
        lo = 0 # the low index of the closest 2 points
        hi = 1 # the highest index of the closest 2 points

        for i in range(m):
            for j in range(i + 1, m):
                dist = 0
                for k in range(n):
                    tmp = pData[i][k] - pData[j][k]
                    dist += tmp * tmp
                    dist = dist * (weight[i] * weight[j]) / (weight[i] + weight[j])

                if dist < min_dist:
                    min_dist = dist
                    lo = i
                    hi = j

        # Merge the 2 minimal distance points
        for k in range(n):
            # incorporate cluster weight
            pData[lo][k] = (pData[lo][k] * weight[lo]) + (pData[hi][k] * weight[hi])
            pData[lo][k] /= (weight[lo] + weight[hi])
        weight[lo] = weight[lo] + weight[hi]

        # copy the last element in the matrix to spot hi - and decrement m by 1
        # because pData[hi] was merged into pData[lo]
        for k in range(n):
            pData[hi][k] = pData[m - 1][k]
        weight[hi] = weight[m - 1]
        # del pData[m - 1]    # delete the last element
        # del weight[m - 1]   # delete the last element
        m -= 1
        # print("pData", pData)
        # print("weight", weight)
        # print("m", m, "k", k)

    # pData will be the cluster centers of the k points

    return pData[:kClusters]

# Do this in numpy
# Then do this with torch to warm up my GPU.
# Do this with minHeap

# Let's do it faster with minHeap, and numpy
from heapq import heapify, heappush, heappop
def KMeansClusterFast(pData, kClusters):  # pData is m X n matrix of samples list of lists, k is the desired number to return
    m, n = pData.shape
    weight = np.ones((m,), dtype=float) # each point is initially weight 1
    print("Start", kClusters, "clusters", type(pData))
    if m < 5:
        print("pData", pData)

    # compute for each point the nearest neighbors
    # find the 2 closest and merge them into a new point
    # the distance should include the approximation introduced by factoring the weight
    # merge points - keep deleting from the end of the array
    heap = []
    heapify(heap)
    if kClusters < m:
        # print("while start m", m, "k", kClusters)
        # find the closest 2 points to merge
        for i in range(m):
            for j in range(i + 1, m):
                dist = 0
                for k in range(n):
                    tmp = pData[i][k] - pData[j][k]
                    dist += tmp * tmp
                    dist = dist * (weight[i] * weight[j]) / (weight[i] + weight[j])

                heappush(heap, (dist, i, weight[i], j, weight[j]))

    while kClusters < m:
        # Merge the 2 minimal distance clusters, the top of the minHeap
        min_dist = heap[0][0]
        lo = heap[0][1]
        weight_lo = heap[0][2]
        hi = heap[0][3]
        weight_hi = heap[0][4]
        heappop(heap)
        # Prune out stuff that got merged already 
        # which is if the current cluster weights don't match what we stored 
        # in minHeap when we computed this distance
        if weight[lo] != weight_lo or weight[hi] != weight_hi:
            # This minimum distance is for 2 clusters that have been merged
            # So skip it, no longer a valid merge option
            continue

        # merge the 2 clusters into the lo one
        for k in range(n):
            # incorporate cluster weight
            pData[lo][k] = (pData[lo][k] * weight[lo]) + (pData[hi][k] * weight[hi])
            pData[lo][k] /= (weight[lo] + weight[hi])
        weight[lo] = weight[lo] + weight[hi]
        # mark the hi cluster as invalid with 0 weight
        weight[hi] = 0.0

        # Find the distance to all the other clusters to this newly formed cluster
        # and place those distances in the minHeap
        for i in range(m):
            # Can't merge with yourself or an invalid cluster
            if lo != i and weight[i] != 0.0:
                dist = 0
                for k in range(n):
                    tmp = pData[i][k] - pData[j][k]
                    dist += tmp * tmp
                    dist = dist * (weight[i] * weight[j]) / (weight[i] + weight[j])

                heappush(heap, (dist, i, weight[i], j, weight[j]))
        # print("pData", pData)
        # print("weight", weight)
        # print("m", m, "k", k)

    # pData will be the cluster centers of the k points

    ans = []
    for i in len(pData):
        if weight[i] != 0.0:            
            ans.append(pData[i])

    return ans

def KMeansTest(matrix, k):
    matrix1 = [matrix[i].copy() for i in range(len(matrix))]
    matrix2 = [matrix[i].copy() for i in range(len(matrix))]
    matrix1 = np.array(matrix1, dtype=float)
    matrix2 = np.array(matrix2, dtype=float)


    time_start = time.time()
    ans1 = KMeansCluster(matrix1, k)
    print(f"{ans1=}")
    time_end1 = time.time()
    ans2 = KMeansClusterFast(matrix2, k)
    time_end2 = time.time()
    time_diff1 = time_end1 - time_start
    time_diff2 = time_end2 - time_end1

    if ans1.tolist() != ans2.tolist():
        print("Error Mismatch")
        print("matrix1", matrix1)
        print("ans1", ans1)
        print("ans2", ans2)
        print("Error Mismatch")
        exit()

    print("ans1", ans1)
    print(f"{time_diff1=:.3f}  {time_diff1=:.3f} seconds.")

matrix = [[1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [2, 2, 2, 2, 2]]
KMeansTest(matrix, 2)
exit()

matrix = [[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[2, 2, 2, 2, 2],
[2, 2, 2, 2, 2]]
KMeansTest(matrix, 1)

matrix = [[4, 5, 6, 7, 2],
[9, 4, 5, 6, 7],
[8, 9, 4, 5, 6],
[3, 8, 9, 4, 5]]
KMeansTest(matrix, 2)

matrix = [[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1]]
KMeansTest(matrix, 2)

matrix = [[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[2, 2, 2, 2, 2]]
KMeansTest(matrix, 2)

matrix = [[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[2, 2, 2, 2, 2]]
KMeansTest(matrix, 1)

matrix = np.random.rand(100,5)
KMeansTest(matrix, 2)

# optimizing ideas
# maintain list of closest points - maybe just 1 closest point - and it's distance
# Just recompute the point distances for the merged point
# Question 2: Check if a matrix is Toeplitz or not

# An example of a Toeplitz matrix is:
# 4 5 6 7 2
# 9 4 5 6 7
# 8 9 4 5 6
# 3 8 9 4 5

 # I would walk along the diagonal from the left side, and check they are all the same till stepping off, walk along the top step down-right and make sure they are all the same


def CheckToeplitz(matrix):
    m = len(matrix) # # of rows
    n = len(matrix[0]) # # of cols

    # walk down left edge
    for i in range(m):  # # of rows
        val = matrix[i][0]
        row = i
        col = 0
        while row < m and col < n:
            if matrix[row][col] != val:
                return False
            row += 1
            col += 1

    # walk across top edge
    for j in range(n):  # # of cols
        val = matrix[0][j]
        row = 0
        col = j
        while row < m and col < n:
            if matrix[row][col] != val:
                return False
            row += 1
            col += 1

    return True


matrix = [[4, 5, 6, 7, 2],
[9, 4, 5, 6, 7],
[8, 9, 4, 5, 6],
[3, 8, 9, 4, 5]]

print("CheckToeplitz Matrix", CheckToeplitz(matrix))

matrix = [[4, 5, 6, 7, 2],
[9, 4, 5, 6, 7],
[8, 9, 4, 5, 6],
[3, 8, 4, 4, 5]]

print("CheckToeplitz Matrix", CheckToeplitz(matrix))


exit()



#                        1 1 1 1 1 1 1 1 1 1
#    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
# 0 |.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|
# 1 |.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|
# 2 |.|.|x|x|x|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|
# 3 |.|.|.|.|.|.|.|.|.|.|.|x|x|x|x|x|x|.|.|.|
# 4 |.|.|.|.|.|x|x|x|.|.|.|x|.|x|.|.|x|.|.|.|
# 5 |.|.|.|.|.|x|.|x|.|.|.|x|x|x|x|x|x|.|.|.|
# 6 |.|.|.|.|.|x|x|x|.|.|.|.|.|.|.|.|.|.|x|x|
# 7 |.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|x|.|
# Assuming a function, count_lakes(image, coord) → integer:
#
# count_lakes(image, (2,2)) → 0
# count_lakes(image, (6,6)) → 1
# count_lakes(image, (12,5)) → 2

def count_lakes(image, coord):
    rowT = len(image)       # total number of row, height of image
    colT = len(image[0])    # total number of col, width of image
    # should error check x,y in rowT,colT and it is land or else return 0
    # DFS from x,y get a set of the points of the island.
    # coord comes in (x,y), code is (row, col) centric, swap order
    island = set([(coord[1], coord[0])])  # the island land pieces we have found
    stack = [(coord[1], coord[0])] 

    # connections include diagonals according to interview instructions
    delta_neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]
    def valid_neighbor(row, col):
        return row >= 0 and row < rowT and col >= 0 and col < colT

    while len(stack) > 0:
        row, col = stack.pop()
        for delta in delta_neighbors:
            rowN = row + delta[0]
            colN = col + delta[1]
            # is it in the image?
            # is it land?
            # Have we not explored it?
            if valid_neighbor(rowN, colN) and \
                image[rowN][colN] == 'x' and  \
                (rowN, colN) not in island:    
                stack.append((rowN, colN)) # explore it's neighbors
                island.add((rowN, colN))   # add to island set so we don't try to explore again

    # Now we have all the land pieces in the island set
    water_visited = set() # keep track of water points visited, so we don't go in circles
    c_lake = 0 # count of lakes found in the island
    # Now enumerate all the land pieces in the island set
    for row, col in island:
        # check if there is connecting water to this land piece
        water_connected = set()
        for delta in delta_neighbors:
            rowN = row + delta[0]
            colN = col + delta[1]
            # is it in the image?
            # is it water?
            # Have we not explored it?
            if valid_neighbor(rowN, colN) and \
                image[rowN][colN] == '.' and \
                (rowN, colN) not in water_visited:
                    water_connected.add((rowN, colN)) # Potential lake
        # Now we have all the water points connected to this land piece
        # Check each one if it's a lake
        for rowWC, colWC in water_connected:
            if (rowWC, colWC) in water_visited:
                continue # water was visited already
            water_visited.add((rowWC, colWC))   # Make sure we never visit it again
            lake_stack = [(rowWC, colWC)]
            is_lake = True
            # get all the water points connected to this 1 point
            # if I touch a image edge, then it's not a lake
            # if I touch land not in my island it's not a lake (island inside a lake case)
            # oops! if I have an island inside my lake inside my island... I'm in trouble!
            # explore all the water points connected to this 1 point before stopping
            # don't break out early or we might think you have a lake when you don't
            # starting from another island point and the offending lake edges were excluded previously
            while len(lake_stack) > 0:
                rowL, colL = lake_stack.pop()
                for delta in delta_neighbors:
                    rowN = rowL + delta[0]
                    colN = colL + delta[1]
                    if not valid_neighbor(rowN, colN):  # did it go off the edge?
                        is_lake = False
                        continue
                    if image[rowN][colN] == 'x' and (rowN, colN) in island:
                        # land on the island - still a lake - skip it
                        continue
                    # else it's land not in my island - treat it like water - really!  Kind of a hack!
                    # if it's an island wrapped around - fine we punch through to the edge and it's not a lake
                    # or if it's an island inside my lake - we punch through and run into the island it is a lake
                    # Both ways work out fine, treating other land as water.
                    # Or it's water in a valid place, better explore it.
                    if (rowN, colN) not in water_visited:
                        lake_stack.append((rowN, colN))
                        water_visited.add((rowN, colN))

            if is_lake == True:
                c_lake += 1

    return c_lake
  

image = [
["|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|"],
["|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|"],
["|.|.|x|x|x|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|"],
["|.|.|.|.|.|.|.|.|.|.|.|x|x|x|x|x|x|.|.|.|"],
["|.|.|.|.|.|x|x|x|.|.|.|x|.|x|.|.|x|.|.|.|"],
["|.|.|.|.|.|x|.|x|.|.|.|x|x|x|x|x|x|.|.|.|"],
["|.|.|.|.|.|x|x|x|.|.|.|.|.|.|.|.|.|.|x|x|"],
["|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|x|.|"]
]

image = [list(x[0].replace("|", "")) for x in image]
for i in image:
  print(i)

print(count_lakes(image, (2,2)), 0)
print(count_lakes(image, (6,6)), 1)
print(count_lakes(image, (12,5)), 2)
print(count_lakes(image, (19,6)), 0)

image = [
["|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|"],
["|x|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|x|"],
["|x|.|x|x|x|.|.|.|.|.|.|.|.|.|.|.|.|.|.|x|"],
["|x|.|.|.|.|.|.|.|.|.|.|x|x|x|x|x|x|.|.|x|"],
["|x|.|.|.|.|x|x|x|.|.|.|x|.|x|.|.|x|.|.|x|"],
["|x|.|.|.|.|x|.|x|.|.|.|x|x|x|x|x|x|.|.|x|"],
["|x|.|.|.|.|x|x|x|.|.|.|.|.|.|.|.|.|.|x|x|"],
["|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|"]
]

image = [list(x[0].replace("|", "")) for x in image]
for i in image:
  print(i)

print(count_lakes(image, (2,2)), 0)
print(count_lakes(image, (6,6)), 1)
print(count_lakes(image, (12,5)), 2)
print(count_lakes(image, (19,6)), 2)

