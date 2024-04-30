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
                    if image[rowN][colN] == 'x':
                        if (rowN, colN) not in island:  # did I touch land not in island?  island in a lake case
                            is_lake = False
                        # else it's land on the island - still a lake
                        continue
                    # It's water in a valid place, better explore it
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

