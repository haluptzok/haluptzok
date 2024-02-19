import time

# Each start/finish state of a list of piles can be represented by a binary field "bin"
# 10 max list members - 10 bits - 2^10 = 1024 is all possible subsets
# "bin" having sBin[index] in it is represented by or-ing in (1 << index), index is 0 to 9 inclusive
# s_bin_to_sum[bin], such that s_sum[3] -> s_sum[00011] -> s_state[0] + s_state[1]

moves_sbin_to_fbin = [[-2]] # Count of split and merge to convert sbin to fbin
s_len = 0              # length of sBin - how many piles in start
s_bin_to_sum_len = 0   # 1 << s_len = 2^s_len = all possible subsets
s_mask = 0             # when inverting values, mask off the extra stuff
s_bin_to_sum = [0]     # binary mask of what values are set, sum up the members
s_sum_to_bin = {}      # Need a has from sum to list of all "bin" that have that sum
f_len = 0
f_bin_to_sum_len = 0
f_mask = 0
f_bin_to_sum = [0]
f_sum_to_bin = {}
bin_to_cMembers = [0] * 1024  # How many members are in the bin  0101 => 2
for i in range(1024):
    i_sum = 0
    for j in range(10):
        if i & (1 << j):
            i_sum += 1
    bin_to_cMembers[i] = i_sum


def splitMergeSmartRec(sBin, fBin):
    global s_len, s_bin_to_sum_len, s_mask, s_bin_to_sum, s_sum_to_bin, f_len, f_bin_to_sum_len, f_mask, f_bin_to_sum, f_sum_to_bin, bin_to_cMembers, moves_sbin_to_fbin

    print("splitMergeSmartRec", bin(sBin), bin(fBin), "sums", s_bin_to_sum[sBin], f_bin_to_sum[fBin])

    if moves_sbin_to_fbin[sBin][fBin] > -2:  # it's been computed
        print("splitMergeSmartRec already computed", moves_sbin_to_fbin[sBin][fBin])
        return moves_sbin_to_fbin[sBin][fBin]

    if s_bin_to_sum[sBin] != f_bin_to_sum[fBin]:
        moves_sbin_to_fbin[sBin][fBin] = -1
        print("splitMergeSmartRec impossible", moves_sbin_to_fbin[sBin][fBin])
        return -1

    # sBin can be broken apart 2^10=1024 ways max - we want find the minimum #
    # for each partition, find all fBin partitions that also work

    cBest = bin_to_cMembers[sBin] - 1 + bin_to_cMembers[fBin] - 1
    # !!! Make a set from enumeration to eliminate duplicates
    # !!! eliminate the ~ also - only do the search if s_bin_part < s_bin_part_inv
    # Can we break sBin in 2 pieces - that also have fBin broken in 2 pieces with matching counts
    for s_bin_part_raw in range(s_bin_to_sum_len):
        # Only want subsets of sBin
        if s_bin_part_raw & sBin != s_bin_part_raw:
            continue
        # We only need the ones that are subsets of sBin
        s_bin_part = s_bin_part_raw
        s_bin_part_inv = (~s_bin_part) & sBin
        # 0 size subsets don't help, have to be breaking piles apart
        # You only have to check it one way
        # s_bin_part_raw & sBin == s_bin_part_raw
        if s_bin_part and s_bin_part_inv and s_bin_part < s_bin_part_inv:
            print(f"{s_bin_part=} {s_bin_part_inv=} {sBin=}", bin(s_bin_part), bin(s_bin_part_inv), bin(sBin))
            # Find all the fBin partitions that match the counts for the sBin partitions
            for f_bin_part in f_sum_to_bin.get(s_bin_to_sum[s_bin_part], []):
                # Is f_bin_part a subset of fBin?
                if f_bin_part == f_bin_part & fBin:
                    f_bin_part_inv = (~f_bin_part) & fBin
                    if True: # Actually have to do both ways -> f_bin_part, f_bin_part_inv:
                        if f_bin_part_inv in f_sum_to_bin.get(s_bin_to_sum[s_bin_part_inv], []):
                            print(f"{f_bin_part=} {f_bin_part_inv=}")
                            cNew_part = splitMergeSmartRec(s_bin_part, f_bin_part)
                            print(f"{cNew_part=}")
                            assert cNew_part > -1
                            cNew_part_inv = splitMergeSmartRec(s_bin_part_inv, f_bin_part_inv)
                            print(f"{cNew_part_inv=}")
                            assert cNew_part_inv > -1
                            if (cNew_part + cNew_part_inv) < cBest:
                                cBest = (cNew_part + cNew_part_inv)
                                print("cBest", cBest)

    print("splitMergeSmartRec returns cBest=", cBest)
    moves_sbin_to_fbin[sBin][fBin] = cBest
    return moves_sbin_to_fbin[sBin][fBin]

def splitMergeSmart(sState, fState):
    global s_len, s_bin_to_sum_len, s_mask, s_bin_to_sum, s_sum_to_bin, f_len, f_bin_to_sum_len, f_mask, f_bin_to_sum, f_sum_to_bin, bin_to_cMembers, moves_sbin_to_fbin
    s_len = len(sState)
    s_bin_to_sum_len = 1 << s_len
    s_mask = (1 << s_len) - 1
    s_bin_to_sum = [0] * s_bin_to_sum_len
    for i in range(s_bin_to_sum_len):
        i_sum = 0
        for j in range(s_len):
            if i & (1 << j):
                i_sum += sState[j]
        s_bin_to_sum[i] = i_sum

    print("splitMergeSmart")
    print(sState)
    print(s_len)
    print(s_bin_to_sum_len)
    print(f"{s_mask=}", s_mask, bin(s_mask))
    print(s_bin_to_sum)

    f_len = len(fState)
    f_bin_to_sum_len = 1 << f_len
    f_mask = (1 << f_len) - 1
    f_bin_to_sum = [0] * f_bin_to_sum_len
    for i in range(f_bin_to_sum_len):
        i_sum = 0
        for j in range(f_len):
            if i & (1 << j):
                i_sum += fState[j]
        f_bin_to_sum[i] = i_sum

    print(fState)
    print(f_len)
    print(f_bin_to_sum_len)
    print(f"{f_mask=}", f_mask, bin(f_mask))
    print(f_bin_to_sum)

    s_sum_to_bin = {} # map a sum of members all the combinations that give that sum
    for i, i_sum in enumerate(s_bin_to_sum):
        print("s i i_sum", i, i_sum)
        new_list = s_sum_to_bin.get(i_sum, []) # get the list for i_sum
        new_list.append(i) # add i into the list
        s_sum_to_bin[i_sum] = new_list
    print(s_sum_to_bin)

    f_sum_to_bin = {} # map a sum of members all the combinations that give that sum
    for i, i_sum in enumerate(f_bin_to_sum):
        print("f i i_sum", i, i_sum)
        new_list = f_sum_to_bin.get(i_sum, []) # get the list for i_sum
        new_list.append(i) # add i into the list
        f_sum_to_bin[i_sum] = new_list
    print(f_sum_to_bin)

    # moves_sbin_to_dbin[sbin][dbin] = # split & merge to get sbin->dbin
    # -2 is uninitialized, -1 if impossible because sums don't match, 0 or more is # of moves
    global moves_sbin_to_fbin
    moves_sbin_to_fbin = [[-2 for _ in range(f_bin_to_sum_len)] for _ in range(s_bin_to_sum_len)]
    # initialize 0 moves for matching start and finish piles
    '''
    !!! figure out why this doesn't work
    for i in range(s_len):
        for j in range(f_len):
            if sState[i] == fState[j]:
                moves_sbin_to_fbin[1 << i][1 << j] = 0  # done with no moves
            else:
                moves_sbin_to_fbin[1 << i][1 << j] = -1 # impossible
    '''
    # Calculate How many moves to get sbin to dbin recursively
    return splitMergeSmartRec((1 << s_len) - 1, (1 << f_len) - 1)

print(splitMergeSmart([1, 2, 3, 4, 5, 6], [7, 7, 7]), 3)
exit()
print(splitMergeSmart([2, 3, 4, 5], [7, 7]), 2)
print(splitMergeSmart([4, 2], [2, 2, 2]), 1)
print(splitMergeSmart([4, 4, 4, 4, 4], [5, 5, 5, 5]), 7)
print(splitMergeSmart([3, 3, 3, 3, 8], [5, 5, 5, 5]), 7)

print(splitMergeSmart([1, 2, 3, 4, 10, 15], [5, 11, 19]), 3)
print(splitMergeSmart([1, 2], [4]), -1)
print(splitMergeSmart([1, 2], [1, 2]), 0)
print(splitMergeSmart([1, 2], [3]), 1)
print(splitMergeSmart([4, 2], [2, 2, 2]), 1)
print(splitMergeSmart([1, 2, 3, 4, 5, 6], [7, 7, 7]), 3)
print(splitMergeSmart([4, 4, 4, 4, 4], [5, 5, 5, 5]), 7)
print(splitMergeSmart([3, 3, 3, 3, 8], [5, 5, 5, 5]), 7)

assert splitMergeSmart([1, 2], [4]) == -1
assert splitMergeSmart([1, 2], [1, 2]) == 0
assert splitMergeSmart([1, 2], [3]) == 1
assert splitMergeSmart([4, 2], [2, 2, 2]) == 1
assert splitMergeSmart([1, 2, 3, 4, 5, 6], [7, 7, 7]) == 3
assert splitMergeSmart([3, 4], [1, 6]) == 2
assert splitMergeSmart([2], [2, 1]) == -1
assert splitMergeSmart([4, 4, 4, 4, 4], [5, 5, 5, 5]) == 7
assert splitMergeSmart([3, 3, 3, 3, 8], [5, 5, 5, 5]) == 7
# assert splitMergeSmart([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], [6, 11, 16, 21, 26, 31, 36, 41, 46, 41], 16)
exit()

# Switch to tuples - need them for caching - does it help/hurt?
# Can I put the lists together better?
# sort the lists - for caching - how much does it hurt?
# Could I cache input output to save time?  Just do 
# check if I am repeating a test - removing 1, 1 - only need to check once
print("hello")
# 1 2 3 4 10 15   -> 5 11 19
# Good 3 merges 1,10->11 4,15->19 2,3->5
# Bad ? merges 1,4->5 2,10->12, 12->11,1, 1->

# Merge together till each merged pile matches a merged pile in the output

worst_moves = 1000  # We know it's never this bad count of moves
max_states = 10 # Max number of stacks

# At 3 points in the code - the best decision is to do the greedy thing and return
# But just in case it wasn't - I searched all other possibilities recursively

old = '''
# For deduping this would be faster in C but slower in python...
    i = 0
    j = 0
    while i < len(startState) and j < len(finishState):
        if startState[i] == finishState[j]:
            del startState[i]
            del finishState[j]
            # don't increment i or j, everything slid down
        elif startState[i] < finishState[j]:
            i += 1
        else:
            # Must be startState[i] > finishState[j]
            j += 1
    '''

def splitMergeRecursive(startState, finishState, be_greedy=False):
    # Remove matching elements between start and finish - can't do better than remove them
    i=0
    while i < len(startState):
        elem = startState[i]
        if elem in finishState:
            startState.remove(elem)
            finishState.remove(elem)
            # don't increment i, everything slid down
        else:
            i += 1

    # Are we done?
    if len(startState) == 0:
        return 0

    cBest = worst_moves # Assume worse than the worst possible

    # Merge 2 input that equal an output
    # Might have to check all options recursively
    for i in range(len(startState)):
        if i > 0 and startState[i] == startState[i - 1]:
            continue
        for j in range(i + 1, len(startState)):
            if j > i + 1 and startState[j] == startState[j - 1]:
                continue
            newelem = startState[i] + startState[j]
            # print("newelem = startState[i] + startState[j]", newelem, startState[i], startState[j])
            if newelem in finishState:
                # Remove the elements and recurse
                startStateCopy = startState.copy()
                finishStateCopy = finishState.copy()
                # print(i, j, startStateCopy, finishStateCopy, startState, finishState)
                finishStateCopy.remove(newelem)
                # print(i, j, startStateCopy, finishStateCopy, startState, finishState)
                del startStateCopy[j]
                # print(i, j, startStateCopy, finishStateCopy, startState, finishState, startState[j])
                del startStateCopy[i]
                # print(i, j, startStateCopy, finishStateCopy, startState, finishState)
                cNew = 1 + splitMergeRecursive(startStateCopy, finishStateCopy) # +1 for the move to merge
                if cNew < cBest:
                    cBest = cNew

    if cBest < worst_moves:  # We already recursed and are done if we found any
        return cBest

    for i in range(len(finishState)):
        if i > 0 and finishState[i] == finishState[i - 1]:
            continue
        for j in range(i + 1, len(finishState)):
            if j > i + 1 and finishState[j] == finishState[j - 1]:
                continue
            newelem = finishState[i] + finishState[j]
            # print("newelem = startState[i] + startState[j]", newelem, startState[i], startState[j])
            if newelem in startState:
                # Remove the elements and recurse
                startStateCopy = startState.copy()
                finishStateCopy = finishState.copy()
                # print(i, j, startStateCopy, finishStateCopy, startState, finishState)
                startStateCopy.remove(newelem)
                # print(i, j, startStateCopy, finishStateCopy, startState, finishState)
                del finishStateCopy[j]
                # print(i, j, startStateCopy, finishStateCopy, startState, finishState, startState[j])
                del finishStateCopy[i]
                # print(i, j, startStateCopy, finishStateCopy, startState, finishState)
                cNew = 1 + splitMergeRecursive(startStateCopy, finishStateCopy) # +1 for the move to merge
                if cNew < cBest:
                    cBest = cNew

    if cBest < worst_moves:  # We already recursed and are done if we found any
        return cBest

    # Split a start pile into 2 so that part of it equals a finish pile
    # Might have to check all options recursively
    if len(startState) < max_states:  # Can't split if we already have max_states
        for i in range(len(startState)):
            if i > 0 and startState[i] == startState[i - 1]:
                continue
            for j in range(len(finishState)):
                if j > 0 and finishState[j] == finishState[j - 1]:
                    continue
                # assert startState[i] != finishState[j] # Should have checked for this already
                if startState[i] > finishState[j]:
                    startStateCopy = startState.copy()
                    finishStateCopy = finishState.copy()
                    startStateCopy[i] -= finishState[j]  # effectively split, and remove the matching one
                    startStateCopy.sort()
                    finishStateCopy.pop(j) # remove the matching one
                    cNew = 1 + splitMergeRecursive(startStateCopy, finishStateCopy) # +1 for the move to split
                    if cNew < cBest:
                        cBest = cNew
                        if be_greedy:
                            return cBest # Actually you can't do any better than this greedy decision

    if cBest < worst_moves:  # We already recursed and are done if we found any
        return cBest

    # Split a finish pile into 2 so that part of it equals a start pile
    # Might have to check all options recursively
    if 0: # if len(finishState) < max_states:  # Can't split if we already have max_states
        for i in range(len(finishState)):
            if i > 0 and finishState[i] == finishState[i - 1]:
                continue
            for j in range(len(startState)):
                if j > 0 and startState[j] == startState[j - 1]:
                    continue
                # assert startState[j] != finishState[i] # Should have checked for this already
                if finishState[i] > startState[j]:
                    startStateCopy = startState.copy()
                    finishStateCopy = finishState.copy()
                    finishStateCopy[i] -= startState[j]  # effectively split, and remove the matching one
                    startStateCopy.pop(j) # remove the matching one
                    cNew = 1 + splitMergeRecursive(startStateCopy, finishStateCopy) # +1 for the move to split
                    if cNew < cBest:
                        cBest = cNew
                        if be_greedy:
                            return cBest # Actually you can't do any better than this greedy decision

    if cBest < worst_moves:  # We already recursed and are done if we found any
        return cBest

    if False:
        # Merge 2 largest piles
        if startState[0] < finishState[0]:
            startState[1] += startState[0]
            del startState[0]
        else:
            finishState[1] += finishState[0]
            del finishState[0]

        # Merge 2 smallest piles
        if startState[0] < finishState[0]:
            startState[1] += startState[0]
            del startState[0]
        else:
            finishState[1] += finishState[0]
            del finishState[0]
    else:
        # We could recurse on all possible pairs, but I don't think we need to
        startState[len(startState) - 2] += startState[len(startState) - 1]
        del startState[len(startState) - 1]

    cBest = 1 + splitMergeRecursive(startState, finishState)
    return cBest

def splitMerge(startState, finishState):
    # Impossible if the sums don't match
    if sum(startState) != sum(finishState):
        return -1
    startState.sort()
    finishState.sort()

    # return splitMergeRecursive(tuple(startState), tuple(finishState))
    return splitMergeRecursive(startState, finishState)

print(splitMerge([1, 2, 3, 4, 10, 15], [5, 11, 19]), 3)
print(splitMerge([1, 2], [4]), -1)
print(splitMerge([1, 2], [1, 2]), 0)
print(splitMerge([1, 2], [3]), 1)
print(splitMerge([4, 2], [2, 2, 2]), 1)
print(splitMerge([1, 2, 3, 4, 5, 6], [7, 7, 7]), 3)
print(splitMerge([4, 4, 4, 4, 4], [5, 5, 5, 5]), 7)
print(splitMerge([3, 3, 3, 3, 8], [5, 5, 5, 5]), 7)

assert splitMerge([1, 2], [4]) == -1
assert splitMerge([1, 2], [1, 2]) == 0
assert splitMerge([1, 2], [3]) == 1
assert splitMerge([4, 2], [2, 2, 2]) == 1
assert splitMerge([1, 2, 3, 4, 5, 6], [7, 7, 7]) == 3
assert splitMerge([3, 4], [1, 6]) == 2
assert splitMerge([2], [2, 1]) == -1
assert splitMerge([4, 4, 4, 4, 4], [5, 5, 5, 5]) == 7
assert splitMerge([3, 3, 3, 3, 8], [5, 5, 5, 5]) == 7
# assert splitMerge([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], [6, 11, 16, 21, 26, 31, 36, 41, 46, 41], 16)

all_tests = '''{1, 2, 3, 4, 10, 15}, {5, 11, 19} 		       3
{5, 1, 2, 3, 4, 10, 15}, {5, 11, 19, 1, 4} 		       4
{5, 11, 19}, {1, 2, 3, 4, 10, 15} 		       3
{5, 11, 19, 8, 8}, {1, 2, 3, 4, 10, 15, 16} 		       4
{5, 11, 19, 8, 4, 4}, {1, 2, 3, 4, 10, 15, 16} 		       5
{1, 2}, {3}		1		
{4, 2}, {2, 2, 2}		1			
{1, 2, 3, 4, 5, 6}, {7, 7, 7}		3			
{3, 4}, {1, 6}		2			
{3, 1, 4, 20, 24, 16, 20}, {5, 15, 21, 14, 16, 12, 2, 3}		5			
{22}, {22}		0			
{26}, {7, 19}		1			
{35}, {14, 11, 10}		2			
{21}, {14, 1, 4, 2}		3			
{5}, {1, 1, 1, 1, 1}		4			
{14}, {2, 4, 2, 2, 3, 1}		5			
{18}, {3, 5, 2, 5, 1, 1, 1}		6			
{14}, {2, 1, 2, 4, 1, 1, 1, 1, 1}		8			
{29, 10}, {31, 8}		2			
{4, 39}, {36, 1, 6}		3			
{30, 31}, {7, 9, 26, 19}		4			
{39, 31}, {23, 19, 5, 15, 8}		3			
{4, 14}, {5, 1, 4, 5, 2, 1}		4			
{13, 15}, {21, 2, 1, 1, 1, 1, 1}		7			
{23, 6, 1, 1, 3, 2, 5, 4}, {39, 6}		6			
{3, 21}, {4, 8, 2, 5, 1, 1, 1, 1, 1}		7			
{6, 3, 32}, {38, 3}		1			
{29, 7, 14}, {28, 10, 12}		4			
{16, 39, 21}, {36, 13, 18, 9}		5			
{25, 39, 1}, {5, 37, 14, 6, 3}		4			
{29, 14, 36}, {13, 30, 18, 6, 5, 7}		5			
{38, 23, 16}, {23, 15, 35, 1, 1, 1, 1}		4			
{25, 6, 23}, {29, 12, 1, 4, 3, 1, 2, 2}		7			
{29, 29, 18}, {37, 3, 6, 6, 4, 16, 2, 1, 1}		8			
{38, 30, 30}, {9, 31, 12, 3, 28, 7, 5, 1, 1, 1}		7			
{14, 35, 5, 12}, {38, 28}		4			
{37, 39, 19, 3}, {31, 42, 25}		3			
{6, 39, 39, 15}, {38, 40, 18, 3}		4			
{11, 38, 30, 3}, {2, 14, 32, 26, 8}		5			
{20, 7, 2, 14}, {3, 27, 5, 6, 1, 1}		4			
{25, 22, 37, 22}, {21, 11, 22, 14, 13, 23, 2}		5			
{11, 13, 5, 28}, {11, 18, 3, 13, 6, 3, 2, 1}		4			
{20, 14, 24, 40}, {4, 25, 23, 5, 3, 28, 5, 1, 4}		7
{28, 23, 25, 6, 10}, {23, 27, 42}		4			
{32, 38, 28, 6, 4}, {45, 21, 35, 7}		5			
{19, 30, 36, 20, 33}, {39, 36, 36, 25, 2}		4			
{28, 30, 7, 32, 19}, {16, 18, 21, 22, 28, 11}		5			
{32, 6, 10, 39, 12}, {3, 25, 17, 5, 34, 4, 11}		6			
{1, 14, 22, 11, 6}, {11, 10, 9, 4, 2, 14, 2, 2}		5			
{28, 6, 29, 17, 35}, {9, 25, 15, 30, 11, 21, 2, 1, 1}		8			
{18, 6, 7, 40, 23, 11}, {41, 28, 32, 4}		6			
{5, 38, 14, 6, 12, 9}, {19, 2, 14, 33, 16}		5			
{14, 38, 26, 17, 6, 26}, {11, 28, 4, 42, 27, 15}		6			
{18, 14, 21, 25, 39, 14}, {11, 26, 20, 10, 20, 26, 18}		7			
{27, 22, 1, 37, 25, 22}, {39, 7, 5, 7, 9, 16, 39, 12}		8			
{40, 5, 38, 2, 20, 28}, {34, 13, 16, 39, 26, 2, 1, 1, 1}		7						
{4, 14, 17, 26, 10, 23, 30}, {41, 36, 47}		4			
{39, 6, 4, 3, 25, 37, 31}, {32, 22, 23, 46, 22}		6			
{3, 21, 40, 11, 18, 25, 30}, {22, 37, 21, 30, 21, 17}		5			
{12, 22, 19, 23, 4, 1, 20}, {6, 19, 22, 22, 14, 5, 13}		4			
{17, 37, 14, 31, 37, 29, 27}, {39, 39, 33, 10, 9, 35, 10, 17}		7			
{3, 29, 22, 36, 27, 27, 38}, {17, 7, 34, 25, 6, 25, 13, 41, 14}		8			
{12, 37, 1, 8, 20, 3, 28}, {38, 24, 39, 1, 1, 2, 1, 1, 1, 1}		9			
{25, 6, 23, 29, 12, 8, 14, 28}, {39, 34, 23, 22, 27}		5			
{20, 14, 24, 40, 4, 25, 23, 5}, {28, 14, 14, 30, 37, 32}		6			
{24, 29, 19, 8, 29, 13, 29, 9}, {4, 20, 37, 36, 30, 21, 12}		7			
{33, 29, 39, 19, 23, 29, 12, 25}, {32, 28, 37, 28, 13, 26, 32, 13}		8			
{20, 21, 6, 25, 12, 27, 30, 40}, {27, 25, 7, 22, 40, 37, 3, 19, 1}		7			
{3, 25, 28, 16, 30, 35, 11, 20}, {23, 23, 27, 29, 39, 2, 17, 1, 4, 3}		8			
{32, 6, 10, 39, 12, 2, 24, 16, 4}, {40, 43, 35, 27}		7			
{1, 14, 22, 11, 6, 11, 14, 1, 24}, {34, 37, 20, 8, 5}		8			
{28, 6, 29, 17, 35, 9, 25, 15, 30}, {45, 12, 20, 35, 40, 42}		7			
{32, 21, 32, 25, 29, 37, 32, 11, 5}, {17, 42, 25, 35, 21, 37, 47}		6			
{1, 21, 4, 36, 14, 13, 14, 35, 33}, {26, 36, 19, 19, 16, 18, 17, 20}		7			
{25, 26, 26, 28, 32, 21, 35, 15, 19}, {22, 20, 22, 16, 12, 47, 48, 14, 26}		8			
{11, 17, 33, 34, 21, 19, 13, 22, 25}, {19, 23, 9, 36, 15, 6, 39, 17, 4, 27}		7			
{28, 13, 12, 2, 4, 11, 33, 2, 30, 19}, {37, 43, 37, 18, 19}		7			
{14, 5, 19, 16, 33, 17, 11, 16, 36, 26}, {27, 25, 43, 40, 33, 25}		6			
{7, 9, 12, 11, 4, 33, 14, 12, 11, 22}, {24, 35, 5, 19, 31, 15, 2, 4}		8			
{16, 10, 32, 23, 30, 10, 4, 36, 31, 30}, {25, 29, 30, 23, 13, 24, 41, 29, 8}		9				
{4, 15, 40, 32, 13, 15, 50, 46, 18, 11}, {27, 33, 19, 23, 12, 39, 34, 35, 20, 2}		10			
{31, 4, 19, 24, 31, 29, 24, 33, 2, 17}, {23, 23, 18, 43, 34, 47, 4, 5, 6, 11}		10			
{8, 2, 35, 11, 47, 32, 41, 12, 4, 28}, {7, 41, 15, 29, 45, 19, 37, 10, 3, 14}		8			
{2, 39, 22, 4, 49, 4, 10, 33, 2, 28}, {13, 28, 6, 10, 20, 15, 7, 28, 27, 39}		6			
{8, 31, 13, 4, 17, 35, 9, 46, 7, 15}, {48, 9, 44, 15, 14, 24, 21, 8, 1, 1}		6			
{5, 29, 37, 14, 33, 7, 34, 46, 42, 32}, {12, 20, 46, 37, 29, 49, 34, 39, 5, 8}		4			
{21, 25, 31, 42, 34, 31, 33, 37, 40, 33}, {49, 28, 38, 45, 28, 47, 36, 8, 12, 36}		12			
{45, 37, 38, 14, 26, 8, 1, 11, 49, 7}, {37, 4, 50, 7, 38, 40, 41, 9, 2, 8}		4			
{13, 9, 31, 45, 33, 9, 18, 31, 44, 46}, {39, 39, 48, 37, 35, 8, 41, 17, 5, 10}		12			
{1}, {2}		-1			
{1}, {1, 1}		-1			
{2}, {2, 1}		-1			
{2, 2}, {1, 2}		-1			
{1, 2, 3}, {3, 1, 1}		-1			
{19, 23, 9, 36, 15, 6, 39, 17, 4, 27}, {27, 4, 17, 39, 6, 15, 36, 9, 23, 19}		0			
{23, 23, 27, 29, 39, 2, 17, 1, 4, 3}, {23, 40, 27, 29, 39, 2, 1, 4, 3}		1			
{23, 23, 27, 29, 39, 2, 17, 1, 4, 3}, {23, 23, 27, 29, 39, 2, 17, 1, 4, 3}		0			
{23, 23, 27, 29, 39, 2, 17, 1, 4, 3}, {23, 23, 27, 29, 39, 3, 17, 1, 4, 3}		-1			
{50, 50, 50, 50, 50, 50, 50, 50, 50, 50}, {50, 50, 50, 50, 50, 50, 50, 50, 50, 50}		0			
{42, 8, 18, 10, 11, 10, 49, 25, 39, 27}, {18, 11, 31, 42, 12, 32, 26, 1, 19, 47}		8			
{1, 2, 3, 4, 6, 20, 24, 25, 25, 26}, {5, 10, 21, 24, 26, 50}		4			
{15, 17}, {3, 4, 10, 15}		2			
{1, 4, 5, 49}, {3, 48, 6, 2}		4			
{42, 35, 20, 29, 13, 6, 32, 12, 46, 28}, {18, 1, 25, 9, 15, 46, 28, 42, 43, 36}		8			
{50, 3, 7, 48, 50}, {30, 20, 12, 46, 50}		4			
{5, 11, 12, 13, 1, 1, 1, 11}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}		6			
{1, 4, 2, 5, 1, 2, 48, 2, 8, 10}, {30, 2, 10, 10, 10, 10, 11}		7			
{11, 12, 13, 14, 15, 16, 17, 18, 19, 20}, {21, 22, 23, 24, 25, 6, 7, 8, 9, 10}		10
{1, 2, 3, 4, 5, 16, 17, 18, 19, 20}, {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}		10
{9, 13, 9, 13, 6, 37}, {25, 37, 10, 8, 1, 2, 1, 1, 1, 1}		8
{18}, {6, 1, 3, 2, 1, 1, 1, 1, 1, 1}		9
{2, 2, 39, 37, 19, 8, 15, 11, 36, 37}, {35, 30, 16, 33, 10, 21, 10, 14, 20, 17}		10		
{24, 29, 19, 8}, {29, 13, 29, 3, 1, 1, 1, 1, 1, 1}		8
{50, 40, 40, 40, 40, 40, 40, 40, 40, 40}, {49, 49, 49, 49, 49, 49, 49, 49, 9, 9}		18
{50, 50, 50}, {15, 15, 15, 16, 14, 13, 17, 15, 10, 20}		9
{32, 21, 32, 25, 29}, {38, 33, 12, 6, 5, 30, 5, 4, 5, 1}		9
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}		0
{2, 2, 2, 4, 4, 50, 50, 8, 12, 2}, {8, 1, 9, 26, 15, 11, 8, 6, 2, 50}		6
{25, 25, 22, 31, 24, 19, 22, 19, 13, 23}, {23, 25, 23, 27, 28, 20, 19, 18, 14, 26}		8
{1, 2, 3, 4, 5, 6}, {7, 7, 7}		3			
{1, 12, 13, 4, 15, 6, 17, 8, 50, 50}, {2, 5, 5, 5, 11, 17, 49, 49, 30, 3}		10			
{12, 22}, {19, 6, 1, 2, 1, 1, 1, 1, 1, 1}		8			
{5, 10, 15, 20, 25, 30, 35, 40, 45, 50}, {6, 11, 16, 21, 26, 31, 36, 41, 46, 41}		16		
'''

# print("all_tests:", all_tests[:30])
print(f"{len(all_tests)=}")
all_tests1 = all_tests.split("\n")
print(f"{len(all_tests1)=}")
total_time_start = time.time()
# for index, line in enumerate(all_tests1[79:80] + all_tests1[116:117] + all_tests1[39:40]):
for index, line in enumerate(all_tests1):
    # print(index, line)
    test = line.strip()
    if len(test) == 0:
        continue
    # print(index, test)
    test = test.split("\t\t")
    arg2 = int(test[1])
    # print(f"{len(test)=}", test)
    args = test[0]
    arg0 = args[(args.find('{') + 1): args.find('}')]
    # print(arg0)
    arg1 = args[len(arg0) + 2:] # .find('{') + 1):]  args.find('}')
    # print(arg1)
    arg1 = arg1[(arg1.find('{') + 1): arg1.find('}')]
    # print(arg1)
    arg0 = arg0.split(',')
    arg0 = [int(x) for x in arg0]
    arg1 = arg1.split(',')
    arg1 = [int(x) for x in arg1]
    # print(arg0)
    # print(arg1)
    # print(arg2)
    time_start = time.time()
    print(index, "splitMerge", arg0, arg1, arg2)
    value = splitMerge(arg0, arg1)
    time_end = time.time()
    time_diff = time_end - time_start
    print(index, "splitMerge", arg0, arg1, arg2, "==", value)
    print(f"Took {time_diff:.3f} seconds {(time_diff/60):.3f} minutes {(time_diff/3600):.3f} hours.\n")
    assert value == arg2
time_end = time.time()
time_diff = time_end - total_time_start
print(f"Total took {time_diff:.3f} seconds {(time_diff/60):.3f} minutes {(time_diff/3600):.3f} hours.\n")

exit()

# int dp[1050][1050];
dp = [[-1] * 1050] * 1050
print("dp len", len(dp), len(dp[0]))
n = 0
m = 0
# int ka0[1050],ka1[1050];
ka0 = [0] * 1050
ka1 = [0] * 1050
a = []
b = []

def rec(x, y):
    global dp,n,m,ka0,ka1,a,b
    # Dynamic programming - Has it been computed already?  Return it
    if dp[x][y] > -1:
        return dp[x][y]
    # If all of x added together and all of y added together - it's 1 to 1 - 0 moves
    if (x==(1<<n)-1 and y==(1<<m)-1):
        return 0
    # ret is worst case estimate
    ret=10000
    # st is the (sum of elements in x) is greater than (sum of elements in y)
    st=ka0[x]-ka1[y]
    if st>0:
        s=1
    else:
        s=0
    for i in range(n): # for every element index i in A
        if (x&(1<<i)): # if x contains that element - skip it
            continue
        ret=min(ret,rec(x+(1<<i),y)+s)  # if x doesn't contain it, solve for x having element i added in
    for j in range(m):
        if (y&(1<<j)): 
            continue
        if (st==b[j]): # if adding element j of B to y makes it's sum equal to x's elements from A 
            ret = min(ret, rec(x,y+(1<<j)))
        if (st>b[j]): 
            ret = min(ret, rec(x,y+(1<<j))+1)
    for i in range(n):
        if (x&(1<<i)):
            continue
        for j in range(m):
            if (y&(1<<j)):
                continue
            # If you add 2 elements that are equal - it takes no moves
            if (a[i]==b[j]):
                ret = min(ret, rec(x+(1<<i),y+(1<<j)))
            # If you add 2 elements that are different size
            elif (a[i]>b[j]):
                ret = min(ret, rec(x+(1<<i),y+(1<<j))+1+s)
            # If you add 2 elements
            elif (a[i]+st==b[j]):
                ret = min(ret, rec(x+(1<<i),y+(1<<j))+1)
            elif (a[i]+st>b[j]):
                ret = min(ret, rec(x+(1<<i),y+(1<<j))+2)

    #print(x, y, ret)
    dp[x][y] = ret
    return ret

def minMoves(A, B):
    global dp,ka0,ka1,a,b,n,m
    n = 0
    m = 0
    # int ka0[1050],ka1[1050];
    ka0 = [0] * 1050
    ka1 = [0] * 1050
    a = A
    b = B
    n=len(a)
    m=len(b)
    # Create every combination of the n members of A summed together 
    for i in range(1 << n):  # power of 2 ^ n 
        for j in range(n):
            if (i&(1<<j)): 
                ka0[i]+=a[j]
    # Create every combination of the m members of B summed together
    for i in range(1 << m):
        for j in range(m):
            if (i&(1<<j)):
                ka1[i]+=b[j]
    # Essentially sum(A) != sum(B) - then it's impossible to do this
    if (ka0[(1<<n)-1]!=ka1[(1<<m)-1]): 
        return -1
    for i in range(1050): 
        for j in range(1050): 
            dp[i][j]=-1;
    return rec(0,0)


# print("all_tests:", all_tests[:30])
print(f"{len(all_tests)=}")
all_tests1 = all_tests.split("\n")
print(f"{len(all_tests1)=}")

for index, line in enumerate(all_tests1[:10]):
    # print(index, line)
    test = line.strip()
    if len(test) == 0:
        continue
    # print(index, test)
    test = test.split("\t\t")
    arg2 = int(test[1])
    # print(f"{len(test)=}", test)
    args = test[0]
    arg0 = args[(args.find('{') + 1): args.find('}')]
    # print(arg0)
    arg1 = args[len(arg0) + 2:] # .find('{') + 1):]  args.find('}')
    # print(arg1)
    arg1 = arg1[(arg1.find('{') + 1): arg1.find('}')]
    # print(arg1)
    arg0 = arg0.split(',')
    arg0 = [int(x) for x in arg0]
    arg1 = arg1.split(',')
    arg1 = [int(x) for x in arg1]
    print(index, "minMoves", arg0, arg1, arg2)
    value = minMoves(arg0, arg1)
    print(index, "minMoves", arg0, arg1, arg2, "==", value)
    # assert value == arg2

#define MP(a,b) make_pair(a,b)
#define REP(i,n) for (int i=0; i<n; ++i)
#define FOREACH(it,x) for(__typeof((x).begin()) it=(x.begin()); it!=(x).end(); ++it)
#define FOR(i,p,k) for (int i=p; i<=k; ++i)
#define PB push_back
#define ALL(x) x.begin(),x.end()
#define SIZE(x) (int)x.size()

# typedef vector <int > VI;
# map<VI,int> mapa[2];
mapa = [{},{}]
# typedef vector<VI> VVI;
# VVI q[2];
q = [ [[]], [[]] ]
# VVI pom;
pom = [[]] # vector of vector of int

def minMoves2(start, end):
    global mapa,q,pom
    start = sorted(start)
    end = sorted(end)
    if sum(start) != sum(end):
        return -1
    if start==end:
        return 0
    
    mapa[0][tuple(start)]=0
    mapa[1][tuple(end)]=0
    q[0].append(start)
    q[1].append(end)
    res=0
    while True:
        res += 1
        for co in range(2): # for (int co=0; co<2; ++co):
            pom = [[]] # list of lists of int
            q[co],pom = pom, q[co] # swap(q[co],pom);
            # FOREACH(it,pom)
            # FOREACH(it,x) for(__typeof((pom).begin()) it=(pom.begin()); it!=(pom).end(); ++it)
            # for(__typeof((pom).begin()) it=(pom.begin()); it!=(pom).end(); ++it)
            for it in pom:
                # VI v=*it;
                v = it.copy()
                n = len(v)
                for i in range(n): # (i=0; i<n; ++i)
                    for j in range(i): # (j=0; j<i; ++j)
                        foo=v.copy()
                        foo[j] += foo[i]
                        foo[i], foo[-1] = foo[-1], foo[i] # swap(foo[i],foo.back());
                        foo.pop()
                        foo = sorted(foo) # sort(ALL(foo));
                        if mapa[1^co].get(tuple(foo), -1) > -1:
                            return res + mapa[1^co][tuple(foo)]
                        if mapa[co].get(tuple(foo),0) == 0:
                            mapa[co][tuple(foo)] = res 
                            q[co].append(foo)
    return -1

# print("all_tests:", all_tests[:30])
print(f"{len(all_tests)=}")
all_tests1 = all_tests.split("\n")
print(f"{len(all_tests1)=}")

for index, line in enumerate(all_tests1[:100]):
    # print(index, line)
    test = line.strip()
    if len(test) == 0:
        continue
    # print(index, test)
    test = test.split("\t\t")
    arg2 = int(test[1])
    # print(f"{len(test)=}", test)
    args = test[0]
    arg0 = args[(args.find('{') + 1): args.find('}')]
    # print(arg0)
    arg1 = args[len(arg0) + 2:] # .find('{') + 1):]  args.find('}')
    # print(arg1)
    arg1 = arg1[(arg1.find('{') + 1): arg1.find('}')]
    # print(arg1)
    arg0 = arg0.split(',')
    arg0 = [int(x) for x in arg0]
    arg1 = arg1.split(',')
    arg1 = [int(x) for x in arg1]
    print(index, "minMoves2", arg0, arg1, arg2)
    value = minMoves2(arg0, arg1)
    print(index, "minMove2", arg0, arg1, arg2, "==", value)

exit()

'''
//SRM307DIV1-1000 SplitAndMergeGame
#include<vector>
#include<cmath>
#include<map>
#include<cstdlib>
#include<iostream>
#include<sstream>
#include<string>
#include<algorithm>
#include<cstring>
#include<cstdio>
#include<set>
#include<stack>
#include<bitset>
#include<functional>
#include<cstdlib>
#include<ctime>
#include<queue>
#include<deque>
using namespace std;
#define pb push_back
typedef long long lint;
#define mp make_pair
#define fi first
#define se second
typedef pair<int,int> pint;
int dp[1050][1050];
int n,m;
int ka0[1050],ka1[1050];
vector <int> a,b;
int rec(int x,int y){
	if(dp[x][y]>-1) return dp[x][y];
	if(x==(1<<n)-1 && y==(1<<m)-1) return 0;
	int ret=10000,st=ka0[x]-ka1[y],i,j,s;
	if(st>0) s=1;else s=0;
	for(i=0;i<n;i++){
		if(x&(1<<i)) continue;ret<?=rec(x+(1<<i),y)+s;
	}
	for(j=0;j<m;j++){
		if(y&(1<<j)) continue;
		if(st==b[j]) ret<?=rec(x,y+(1<<j));if(st>b[j]) ret<?=rec(x,y+(1<<j))+1;
	}
	for(i=0;i<n;i++){
		if(x&(1<<i)) continue;
		for(j=0;j<m;j++){
			if(y&(1<<j)) continue;
			if(a[i]==b[j]) ret<?=rec(x+(1<<i),y+(1<<j));
			else if(a[i]>b[j]) ret<?=rec(x+(1<<i),y+(1<<j))+1+s;
			else if(a[i]+st==b[j]) ret<?=rec(x+(1<<i),y+(1<<j))+1;
			else if(a[i]+st>b[j]) ret<?=rec(x+(1<<i),y+(1<<j))+2;
		}
	}
//	cout<<x<<' '<<y<<' '<<ret<<endl;
	return dp[x][y]=ret;
}
class SplitAndMergeGame{
	public:
	int SplitAndMergeGame::minMoves(vector <int> A,vector <int> B){
		int i,j;
		a=A;b=B;n=a.size();m=b.size();memset(ka0,0,sizeof(ka0));memset(ka1,0,sizeof(ka1));
		for(i=0;i<(1<<n);i++) for(j=0;j<n;j++){if(i&(1<<j)) ka0[i]+=a[j];}
		for(i=0;i<(1<<m);i++) for(j=0;j<m;j++){if(i&(1<<j)) ka1[i]+=b[j];}
		if(ka0[(1<<n)-1]!=ka1[(1<<m)-1]) return -1;
		for(i=0;i<1050;i++) for(j=0;j<1050;j++) dp[i][j]=-1;
		return rec(0,0);
	}
};




#include <cstdio>
#include <iostream>
#include <algorithm>
#include <set>
#include <map>
#include <stack>
#include <list>
#include <queue>
#include <deque>
#include <cctype>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <numeric>
using namespace std;
 
#define MP(a,b) make_pair(a,b)
 
typedef vector <int > VI;
typedef vector<VI> VVI;
#define REP(i,n) for (int i=0; i<n; ++i)
#define FOREACH(it,x) for(__typeof((x).begin()) it=(x.begin()); it!=(x).end(); ++it)
#define FOR(i,p,k) for (int i=p; i<=k; ++i)
#define PB push_back
#define ALL(x) x.begin(),x.end()
#define SIZE(x) (int)x.size()
 
map<VI,int> mapa[2];
VVI q[2];
VVI pom;
 
    class SplitAndMergeGame
        { 
        public: 
        int minMoves(vector <int> start, vector <int> end){ 
              sort(ALL(start));
              sort(ALL(end));
              if (accumulate(ALL(start),0)!=accumulate(ALL(end),0)) return -1;
              if (start==end) return 0;
              
              mapa[0][start]=0; mapa[1][end]=0;
              q[0].PB(start); q[1].PB(end);
              int res=0;
              while (1){
                res++;
                REP(co,2){
                  pom.clear(); swap(q[co],pom);
                  FOREACH(it,pom){
                    VI v=*it;
                    int n=SIZE(v);
                    REP(i,n) REP(j,i){
                      VI foo=v;
                      foo[j]+=foo[i];
                      swap(foo[i],foo.back());
                      foo.pop_back();
                      sort(ALL(foo));
                      if (mapa[1^co].count(foo)) return res+mapa[1^co][foo];
                      if (mapa[co].count(foo)==0){
                        mapa[co].insert(MP(foo,res)); q[co].PB(foo);
                      }
                    }
                  }         
                } 
              }
              return -1; 
            } 
        
 
         }; 
 
    

Definition	
Class:	SplitAndMergeGame
Method:	minMoves
Parameters:	int[], int[]
Returns:	int
Method signature:	int minMoves(int[] startState, int[] finishState)
(be sure your method is public)
Constraints
-	startState will contain between 1 and 10 elements, inclusive.
-	finishState will contain between 1 and 10 elements, inclusive.
-	Each element of startState will be between 1 and 50, inclusive.
-	Each element of finishState will be between 1 and 50, inclusive.
Examples
0)	
    	
{1, 2}
{3}
Returns: 1
Merge the two piles to form a single pile of 3 coins.


1)	
    	
{4, 2}
{2, 2, 2}
Returns: 1
Split the pile of 4 coins into two piles of 2 coins.

2)	
    	
{1, 2, 3, 4, 5, 6}
{7, 7, 7}
Returns: 3

3)	
    	
{3, 4}
{1, 6}
Returns: 2
One way to do this is to split the pile of 3 coins into a pile of 2 coins and a pile with 1 coin. Then, merge the pile of 2 coins with the pile of 4 coins to form a pile of 6 coins.


4)	
    	
{2}
{2,1}
Returns: -1
A solution doesn't exist.


def splitMerge(startState, finishState):
	# check it is possible - sum(startState) == sum(finishState)

	# Easiest solution: combine all into n, split n-1,1), keep adding right side to iterate through the final state goals
# combine all into n, split out the desired final state solutions
# cntInput
# cntOutput
# moves = (cInput - 1) + (cntOutput - 1)

# greedy smarter - that might not be optimal
# eliminate piles from start/finish that are already correct
# any input piles to merge that equal output piles - do it and remove
# split to do any output
# merge input


# search approach
# smart and remove any start==finish sized piles

Problem Statement			    	  	 	 	   	
Split-and-merge is a one player game. The player starts out with several piles of coins. With each move, he can either merge two of the piles into a single pile, or split a single pile into two new non-empty piles. You are given a int[] startState, containing the starting configuration of the coins, and a int[] finishState, containing the target configuration. Each element of the int[]s represents the number of coins in a pile. The order of the elements do not matter. For example, {1, 2, 3} and {2, 1, 3} represent the same set of piles. Return the minimal number of moves necessary to reach the finishState from the startState. If a solution doesn't exist then return -1.
Definition	
Class:	SplitAndMergeGame
Method:	minMoves
Parameters:	int[], int[]
Returns:	int
Method signature:	int minMoves(int[] startState, int[] finishState)
(be sure your method is public)
Constraints
-	startState will contain between 1 and 10 elements, inclusive.
-	finishState will contain between 1 and 10 elements, inclusive.
-	Each element of startState will be between 1 and 50, inclusive.
-	Each element of finishState will be between 1 and 50, inclusive.
Examples
0)	
    	
{1, 2}
{3}
Returns: 1
Merge the two piles to form a single pile of 3 coins.


1)	
    	
{4, 2}
{2, 2, 2}
Returns: 1
Split the pile of 4 coins into two piles of 2 coins.

2)	
    	
{1, 2, 3, 4, 5, 6}
{7, 7, 7}
Returns: 3

3)	
    	
{3, 4}
{1, 6}
Returns: 2
One way to do this is to split the pile of 3 coins into a pile of 2 coins and a pile with 1 coin. Then, merge the pile of 2 coins with the pile of 4 coins to form a pile of 6 coins.


4)	
    	
{2}
{2,1}
Returns: -1
A solution doesn't exist.


def splitMerge(startState, finishState):
	# check it is possible - sum(startState) == sum(finishState)

	# Easiest solution: combine all into n, split n-1,1), keep adding right side to iterate through the final state goals
# combine all into n, split out the desired final state solutions
# cntInput
# cntOutput
# moves = (cInput - 1) + (cntOutput - 1)

# greedy smarter - that might not be optimal
# eliminate piles from start/finish that are already correct
# any input piles to merge that equal output piles - do it and remove
# split to do any output
# merge input


# search approach
# smart and remove any start==finish sized piles



'''


'''
'python imbue.py' will print out results for all examples 
(using python11, less than python 3.6 won't run this code)

Code Summary:

1> count_checkpoints - Find all the N checkpoints
2> compute_dist_matrix - Compute NXN matrix of distance & paths between all checkpoints with Djikstras
3> enumerate the subsets of nodes ~(N choose K) - K of the checkpoint nodes
    4a> compute_min_length_for_subset - Find min path length for K! possible permutation ordering (too slow for examples 4&5)
    4b> compute_min_length_for_subset_fast - Find min path by greedily inserting checkpoint node maximally distant from current path (does ex 1 to 4)
    4c> compute_covered_path - plot out each kXk paths - count squares occuped - subtract out max path distance in kXk (does ex 1 to 4)

5> approximate sampling solution - enumerate randomly 256 of the (N choose K) subsets - return average

6> Not my solution - convert java code to python from correct solution on topcoder
    - included for fun / comparison
    - this solution does ex 1-5 exactly - mine does 1-4 exactly - mine only can do 5 approximately
    - added comments / explanation
    - no way I could come with this solution in 2 hours - it's impressive!
'''
import time
import numpy as np
import itertools
from typing import List
import math
import random

def count_checkpoints(field):
    l_return = []
    i_line = 0
    for line in field:
        i_char = 0
        for char in line:
            if char == "*":
                l_return.append([i_line, i_char])
            i_char += 1
        i_line += 1
    return l_return

# given a path of checkpoints, compute the length of it
def compute_path_length(path, dist_matrix):
    dist = 0
    
    for i in range(len(path) - 1):
        dist += dist_matrix[path[i]][path[i+1]]

    return dist

# compute_min_length_for_subset - for a subset of size K checkpoint nodes
# compute the minimal path visiting all the nodes by computing 
# all K! permutations of the checkpoint node order and computing that length
# Return the minimum path length from all the permutations tried
from itertools import permutations
def compute_min_length_for_subset(field, subset, dist_matrix):
    # For a subset compute all permutations
    list_set = list(permutations(subset))
    # len_path is K, go through all checkpoints in the subset
    len_path = len(subset)
    # For each permuation sum up the path distance from from node to node
    min_dist = len(field) * len(field[1]) * len(subset) # width * height * K is worst case
    for path in list_set:
        # Remove half that are just the same path backwards - path costs are the same each way.
        # Remove half by making sure first index < last index - don't need to compute both directions.
        if path[0] < path[len_path - 1]:
            path_dist = compute_path_length(path, dist_matrix)
            if path_dist <= min_dist:
                min_dist = path_dist

    return min_dist

# compute_min_length_for_subset_fast - for a subset of size K checkpoint nodes
# compute the minimal path by greedily inserting the furthest checkpoint to the existing
# path until all checkpoints are in, inserting where they are the shortest length
def compute_min_length_for_subset_fast(field, subset, dist_matrix):
    c_checkpoint = len(dist_matrix)
    pts_in_path = np.zeros(c_checkpoint, dtype=int) # which points are in the path

    dist_max = 0
    row_max = 0
    col_max = 0

    # Find the longest path between to start with
    for row in subset:
        for col in subset:
            if dist_matrix[row][col] > dist_max:
                dist_max = dist_matrix[row][col]
                row_max = row
                col_max = col

    min_path = [row_max]  # build the shortest path through all the checkpoints
    path_len = 1          # keep track of how many points are in the path
    pts_in_path[row_max] = 1   # it's in the path now

    while path_len < len(subset):
        # find the next checkpoint to add into the path
        # pick the checkpoint farthest from the points in path
        dist_max = 0
        row_max = 0
        col_max = 0
        for row in min_path:
            for col in subset:
                if dist_matrix[row][col] > dist_max and pts_in_path[col] == 0: # col not in min_path:
                    dist_max = dist_matrix[row][col]
                    row_max = row
                    col_max = col

        dist_min_new_path = 100 + len(field) * len(field[1]) # worst case path length
        best_new_path = None
        for i in range(path_len + 1):
            new_path = min_path.copy()
            new_path.insert(i, col_max)
            dist_new_path = compute_path_length(new_path, dist_matrix)
            if dist_new_path < dist_min_new_path:
                best_new_path = new_path
                dist_min_new_path = dist_new_path

        # Put the point in where it's shortest
        min_path = best_new_path
        path_len += 1
        pts_in_path[col_max] = 1 # it's in the path now

    return dist_min_new_path

# returns checkpoint_dist_matrix - distance between all checkpoint node
# checkpoint_dist_matrix[i][j] is the distance from checkpoint i to j
# Use Djikstra's algo to compute dist from all checkpoints
# returns checkpoint_path_matrix - paths between all checkpoint node
# checkpoint_path_matrix[i][j] is the path from checkpoint i to j
from collections import deque
def compute_dist_matrix(l_checkpoints, field):
    checkpoint_dist_matrix = []
    checkpoint_path_matrix = []
    c_row = len(field)
    print(f"{c_row=}")    
    c_col = len(field[0])
    print(f"{c_col=}")
    shape_matrix = [c_row, c_col]
    size_matrix = c_row * c_col
    
    # compute distance from checkpoint_start to every checkpoint in l_checkpoints using Djistra's algorithm
    for checkpoint_start in l_checkpoints:
        # Init a c_row X c_col matrix to hold the dist of min path
        field_dist_mat = np.ones(shape_matrix, np.int32)  # create array of distances from checkpoint_start
        field_dist_mat *= size_matrix # initialize with estimate that is +1 bigger than worst case path of visiting every node once
        # Init a c_row X c_col matrix to hold the min paths
        field_path_mat = [[[] for col in range(c_col)] for row in range(c_row)]

        q = deque() # queue of nodes to explore
        q.append(checkpoint_start) # start with source node of cost 0
        field_dist_mat[checkpoint_start[0], checkpoint_start[1]] = 0 # distance start to start is zero
        field_path_mat[checkpoint_start[0]][checkpoint_start[1]] = [[checkpoint_start[0],checkpoint_start[1]]]

        # while there are nodes left to explore
        while len(q) > 0:
            node = q.popleft()
            row = node[0]
            col = node[1]
            new_cost = field_dist_mat[row, col] + 1

            # explore up
            if row > 0:
                if field_dist_mat[row - 1, col] > new_cost and field[row - 1][col] != '#':
                    field_dist_mat[row - 1, col] = new_cost
                    field_path_mat[row - 1][col] = field_path_mat[row][col] + [[row - 1, col]]
                    q.append([row - 1, col]) # add to list of nodes to explore

            # explore down
            if row < c_row - 1:
                if field_dist_mat[row + 1, col] > new_cost and field[row + 1][col] != '#':
                    field_dist_mat[row + 1, col] = new_cost
                    field_path_mat[row + 1][col] = field_path_mat[row][col] + [[row + 1, col]]
                    q.append([row + 1, col]) # add to list of nodes to explore

            # explore left
            if col > 0:                
                if field_dist_mat[row, col - 1] > new_cost and field[row][col - 1] != '#':
                    field_dist_mat[row, col - 1] = new_cost
                    field_path_mat[row][col - 1] = field_path_mat[row][col] + [[row, col - 1]]
                    q.append([row, col - 1]) # add to list of nodes to explore

            # explore right
            if col < c_col - 1:
                if field_dist_mat[row, col + 1] > new_cost and field[row][col + 1] != '#':
                    field_dist_mat[row, col + 1] = new_cost
                    field_path_mat[row][col + 1] = field_path_mat[row][col] + [[row, col + 1]]
                    q.append([row, col + 1]) # add to list of nodes to explore

        # make a list of all the dists and paths to append
        row_dist = []
        row_path = []
        for checkpoint_end in l_checkpoints:
            row_dist.append(field_dist_mat[checkpoint_end[0], checkpoint_end[1]])
            row_path.append(field_path_mat[checkpoint_end[0]][checkpoint_end[1]])

        checkpoint_dist_matrix.append(row_dist)
        checkpoint_path_matrix.append(row_path)
    
    return checkpoint_dist_matrix, checkpoint_path_matrix

def max_length_path_for_subset(field, dist_matrix, subset):
    max_length_path = 0

    for i_cp1 in range(len(subset)):
        for i_cp2 in range(i_cp1 + 1, len(subset)):
            length_path = dist_matrix[subset[i_cp1]][subset[i_cp2]]
            if length_path > max_length_path:
                max_length_path = length_path

    return max_length_path

def compute_covered_squares(field, dist_matrix, path_matrix, subset):
    # Plot out all the paths in the subset of checkpoints into a matrix
    # print(f"{field=}")
    # print(f"{dist_matrix=}")
    # print(f"{path_matrix=}")
    # print(f"{subset=}")
    c_row = len(field)
    c_col = len(field[0])
    shape_matrix = [c_row, c_col]
    field_plot_mat = np.zeros(shape_matrix, np.int32)  

    for i_cp1 in range(len(subset)):
        for i_cp2 in range(i_cp1 + 1, len(subset)):
            path = path_matrix[subset[i_cp1]][subset[i_cp2]]
            # print(f"{path=}")
            for point in path:
                field_plot_mat[point[0], point[1]] = 1

    # Sum up how many positions are occupied
    return np.sum(field_plot_mat)

def expected_length(field: List[str], K: int) -> float:
    # Get a list of all the checkpoint coordinates
    l_checkpoints = count_checkpoints(field)  
    # Count how many checkpoint coordinates there are
    c_checkpoints = len(l_checkpoints) 
    print(f"{c_checkpoints=} {K=}")
    # Check how many subsets - is enumeration viable?
    print("number of subsets is", math.comb(c_checkpoints,K))
    expected_length_ret = 0
            
    # First compute dist_matrix with djikstas algo - from each checkpoint node to all others.
    # dist_matrix[i][j] is the distance from checkpoint index i to j - compute once for speed
    # path_matrix[i][j] is the path from checkpoint index i to j - compute once for speed
    time_start = time.time()
    dist_matrix, path_matrix = compute_dist_matrix(l_checkpoints, field)
    time_end = time.time()
    time_diff = time_end - time_start
    print(f"dist_matrix took {time_diff:.3f} seconds.")

    # 4a> First Approach:
    # Naive/Simple - but implemented first to test compute_dist_matrix worked properly.
    # If K is small - do (n choose k) subsets, for each subset judge k! permutations of nodes
    # sum the minimum cost permutation's path cost, for all the (n choose k) subsets.
    # This is slow - it's brute force and what I first thought of for traveling salesman like solution.
    # It works fine for first 3 examples - too slow for last 2 bigger examples.
    # Left here just to show it works for small examples - verifies compute_dist_matrix is all good
    if K < 9:
        time_start = time.time()
        i_checkpoints = list(range(c_checkpoints)) # array of indexes into l_checkpoints to take subsets of

        iter_checkpoints_subset = itertools.combinations(i_checkpoints, K)
        c_subsets = 0
        total_len = 0
        # for each subset of checkpoints of size K, compute the minimal path to visit them all
        # Compute the sum of all such paths to compute the average expected value for randomly choosing 2 points
        for subset in iter_checkpoints_subset:
            c_subsets += 1
            total_len += compute_min_length_for_subset(field, subset, dist_matrix)

        expected_length_ret = total_len / c_subsets
        print(f"{c_subsets=}")
        print(f"{total_len=}")
        print("compute_min_length_for_subset computes", expected_length_ret)
        time_end = time.time()
        time_diff = time_end - time_start
        print(f"compute_min_length_for_subset took {time_diff:.3f} seconds.")
    # #2 Second Approach
    # For complexity we have a product of 2 exponentials (n choose k) and k! - yuck!
    # My first thought was to get rid of the k!
    # To visit all checkpoints and return to the start, because of the 4-connected property,
    # which says there is only 1 path to/from any pair of checkpoints then min the path length would be:
    # Shortest_Length_Path = (2 * (squares_occupied - 1)) -> Because we transition between every node twice.
    # 
    # But we don't have to return to the start - so the shortest path to visit all K checkpoints is 
    # (2 * (squares_occupied_following_all_KXK_connecting_paths - 1)) - 
    # (length of the longest path between any 2 of the K Nodes)
    # 
    # So still do (n choose k) subsets, pick minimum cost by greedily picking checkpoints to add to the path.
    # We can start by adding the maxmimum length path in the table between the active K checkpoints.
    # Then keep adding the maximum length path to missing checkpoint nodes to the path, till they are all added,
    # You essentially keep adding side paths off the longest path, the side paths get traversed twice, out and 
    # back, so you continue on the longest path that you don't have to traverse back.
    if K < 150:
        time_start = time.time()
        i_checkpoints = list(range(c_checkpoints)) # array of indexes into l_checkpoints to take subsets of
        iter_checkpoints_subset = itertools.combinations(i_checkpoints, K)
        c_subsets = 0
        total_len = 0
        # for each subset of checkpoints of size K, compute the minimal path to visit them all
        # Compute the sum of all such paths to compute the average expected value for randomly choosing 2 points
        for subset in iter_checkpoints_subset:
            c_subsets += 1
            total_len += compute_min_length_for_subset_fast(field, subset, dist_matrix)

        expected_length_ret = total_len / c_subsets
        print(f"{c_subsets=}")
        print(f"{total_len=}")
        print("compute_min_length_for_subset_fast computes", expected_length_ret)
        time_end = time.time()
        time_diff = time_end - time_start
        print(f"compute_min_length_for_subset_fast took {time_diff:.3f} seconds.")

    # If you were to visit all checkpoint nodes and return to the start checkpoint
    # you would visit 2 * (count_vertex - 1).  But since you don't go back to the start, the shortest path
    # to visit all the nodes includes the longest path from any of the K checkpoints - because you have to 
    # traverse every vertex twice if you return to the start, but if you don't have to return to the start
    # then having the longest path from the start node to the end node means you visit:
    # 2 * (count_vertex - 1) - path_from_startK_to_endK

    # Just plot the k^2 paths to count up the occupied squares
    # double that and subtract out the longest path which you don't have to traverse back
    if K < 150:
        time_start = time.time()
        i_checkpoints = list(range(c_checkpoints)) # array of indexes into l_checkpoints to take subsets of
        iter_checkpoints_subset = itertools.combinations(i_checkpoints, K)
        c_subsets = 0
        total_len = 0
        # for each subset of checkpoints of size K, compute the minimal path to visit them all
        # Compute the sum of all such paths to compute the average expected value for randomly choosing 2 points
        c_covered_squares_total = 0
        max_path_len_total = 0

        for subset in iter_checkpoints_subset:
            c_subsets += 1
            c_covered_squares = compute_covered_squares(field, dist_matrix, path_matrix, subset)
            c_covered_squares_total += c_covered_squares
            # print(f"{c_covered_squares=}")
            total_len += 2 * (c_covered_squares - 1)
            max_path_len = max_length_path_for_subset(field, dist_matrix, subset)
            max_path_len_total += max_path_len
            # print(f"{max_path_len=}")
            total_len -= max_path_len

        print(f"{c_covered_squares_total=} {max_path_len_total=}")
        print(f"{c_covered_squares_total/c_subsets=} {max_path_len_total/c_subsets=}")

        print(f"{c_subsets=}")
        print(f"{total_len=}")
        expected_length_ret = total_len / c_subsets
        print("covered_squares - max_len_path gives:", expected_length_ret)
        expected_length_ret = (((2 * c_covered_squares_total) - max_path_len_total) / c_subsets) - 2
        print("covered_squares - max_len_path gives:", expected_length_ret)

        time_end = time.time()
        time_diff = time_end - time_start
        print(f"covered_squares took {time_diff:.3f} seconds.")

    # Sampling approach - randomly choose total_samples of K of the N checkpoints
    # Compute approximate by sampling the subsets of (N choose K)
    # This works for all examples 1 to 5, so always do it to show the result for comparison
    if True:
        # make it repro results while debugging with fixed rand seeds
        random.seed(42)
        np.random.seed(42)        
        time_start = time.time()
        a_checkpoints = np.arange(0, c_checkpoints) # array of indexes into l_checkpoints to take subsets of
        total_samples = 256
        total_len = 0
        c_covered_squares_total = 0
        max_path_len_total = 0
        # for each random subset of checkpoints of size K, compute the minimal path to visit them all
        for _ in range(total_samples):
            subset = np.random.choice(a_checkpoints, K, replace=False)
            if False:
                new_len = compute_min_length_for_subset_fast(field, subset, dist_matrix)
                total_len += new_len
            else:
                c_covered_squares = compute_covered_squares(field, dist_matrix, path_matrix, subset)
                c_covered_squares_total += c_covered_squares
                # print(f"{c_covered_squares=}")
                total_len += 2 * (c_covered_squares - 1)
                max_path_len = max_length_path_for_subset(field, dist_matrix, subset)
                max_path_len_total += max_path_len
                # print(f"{max_path_len=}")
                total_len -= max_path_len

        expected_length_ret = total_len / total_samples
        print("sampling covered_squares - max_len_path gives:", expected_length_ret)
        expected_length_ret = (((2 * c_covered_squares_total) - max_path_len_total) / total_samples) - 2
        print("sampling covered_squares - max_len_path gives:", expected_length_ret)
                
        expected_length_ret = total_len / total_samples
        print(f"{total_samples=}")
        print(f"{total_len=}")
        print(f"sampling {total_samples} compute_min_length_for_subset_fast computes", expected_length_ret)

        time_end = time.time()
        time_diff = time_end - time_start
        print(f"sampling took {time_diff:.3f} seconds.")

    # My approach won't make Example 5 run - (300 choose 150) is too big
    # Need to approach it differently to eliminate the (n choose k) exponential issue.
    # Google gave me this solution https://community.topcoder.com/stat?c=problem_solution&cr=10574855&rd=15183&pm=12305
    # It's in Java and all the variable names are a,b,c 
    # So I converted it to Python
    # and then it was non-trivial to understand and explain it below.
    # But it was fun!

    # Vertex is created for each passable square
    class Vertex:
        def __init__(self):
            self.generation = 0  # int - temporary version id used for doing Djikstas
            self.distance = 0    # int - temporary distance used for doing Djikstras
            self.index = 0       # int - index of Vertex
            self.checkpoint = 0  # int, 1 if checkpoint, 0 otherwise
            self.adj_edge = []   # List of actual adjacent edges - 4 max (left,right,up,down)

    # Create an edge for each valid transition from adjacent squares in the field
    class Edge:
        def __init__(self, dest):
            self.vertex_dest = dest     # The vertex the edge leads to
            self.ai_checkpointDistMap = []  # array of distances to all other checkpoints from this square
            self.i_count = 0             # int
            self.i_savedCount = 0        # int

    lastGen = 0
    height = len(field)
    width = len(field[0])
    # Create matrix correpsonding to field with a Vertex for each passable square (. or *)
    mat_vertex = [[None for _ in range(width)] for _ in range(height)]
    totalCheckpoints = 0
    totalVertices = 0
    arr_all_vertex = []

    # This is a Djikstra's flavor computation of an array of distances
    # to checkpoints from a square for each of the edges.
    # Each index is the returned array (index 0 is 1 step) is 
    # the count of checkpoints at that distance from this square on that
    # edge.  Note it doesn't count itself - all other n-1 checkpoints besides 
    # itself.  And it might be just a passable square so then the distance
    # to the N other checkpoints.
    # So forbidden is the square you are on - blocking you to only explore on 
    # the "side" of the square for this edge.
    # I note they use a queue to put squares in to explore, but don't
    # bother popping them off the left, just incrementing the index qt to access
    # the head of the queue.  It all gets freed on return from the function
    # so it works fine and just uses more memory to speed it up a little.
    # Also note generation is just used to "reset" the Djikstras to let you know if the
    # distances are valid - I guess cheaper than giving everyone a worst-case distance
    # every time you restart the Djiksta's search
    def countCheckpoints(root, forbidden):
        nonlocal lastGen
        nonlocal totalVertices
        queue = []
        qt = 0
        queue.append(root)
        res = [0] * totalVertices
        lastGen += 1
        forbidden.generation = lastGen
        root.generation = lastGen
        root.distance = 0
        while qt < len(queue):
            cur = queue[qt]
            qt += 1
            if cur.checkpoint:
                res[cur.distance] += 1
            for e in cur.adj_edge:
                a = e.vertex_dest
                if a.generation < lastGen:
                    a.generation = lastGen
                    a.distance = cur.distance + 1
                    queue.append(a)
        return res

    # compute mat_vertex[height][width] with a vertex for every passable square (not #)
    # Also compute arr_all_vertex which is an array of all the passable squares which 
    # is slightly easier to emuerate through than just walking through the matrix and checking
    # for a vertex not being None.
    for r in range(height):
        for c in range(width):
            if field[r][c] != '#':
                mat_vertex[r][c] = Vertex()
                mat_vertex[r][c].index = totalVertices                
                if field[r][c] == '*':
                    mat_vertex[r][c].checkpoint = 1
                    totalCheckpoints  += 1
                totalVertices += 1
                arr_all_vertex.append(mat_vertex[r][c])

    print(f"{totalCheckpoints=} {totalVertices=}")

    # For each vertex in field (a vertex is both * and . so any passable square)
    # calculate from the 4 possible edges out (left, right, up, down) how far away 
    # each of the checkpoints are from it, so index 0 is 1 move, and each edge
    # has an array of counts of how many at each distance, 
    # distance is max limited to 50*50=2500 if the path went through every square
    # so practically less than half of that. (50X50 is max field size)
    # All the other checkpoints doesn't include yourself - so for a '.' square 
    # the total sum of the array is N, or for a checkpoint the total is N-1.
    for r in range(height):        
        for c in range(width):
            if mat_vertex[r][c] != None:
                if r + 1 < height and mat_vertex[r + 1][c] != None:
                    mat_vertex[r][c].adj_edge.append(Edge(mat_vertex[r + 1][c]))
                    mat_vertex[r + 1][c].adj_edge.append(Edge(mat_vertex[r][c]))
                if c + 1 < width and mat_vertex[r][c + 1] != None:
                    mat_vertex[r][c].adj_edge.append(Edge(mat_vertex[r][c + 1]))
                    mat_vertex[r][c + 1].adj_edge.append(Edge(mat_vertex[r][c]))

    # This "c" array is computing a table of binomial coefficients for efficient
    # look up of (n choose k).  The so each row index i is (x + y)^i and each
    # column j is the index into the coeffients.
    # Example row 0 is (x+y)^0 -> 1 -> [1 0 0 0 ...]
    # Example row 2 is for (x+y)^2 -> 1*x^2 + 2*x*y + 1* -> [1 2 1 0 0 0 ...]
    # So computing (N choose K) = c[N][K]
    c = np.zeros([totalCheckpoints + 1, totalCheckpoints + 1], np.double)
    c[0][0] = 1.0
    for i in range(1, totalCheckpoints + 1):
        c[i][0] = 1.0
        for j in range(1, totalCheckpoints + 1):
            c[i][j] = c[i - 1][j - 1] + c[i - 1][j]

    # print(f"{c=}")

    # res is initialized with the count of moves to walk to every 
    # vertex/square and back to where you started - all of the squares (* and .)
    res = 2.0 * (totalVertices - 1)
    # for each vertex/square compute the expected number of times 
    # it won't be included in the active paths.
    # So imagine a square '.' off the the edge with N checkpoints on it's right.
    # It will always be excluded - (N choose K) permutations on the right of it
    # out of (N choose K) total permutations -->  c[cntHere][K] / c[totalCheckpoints][K] == 1
    # Or if a square has N/2 on left side and N/2 checkpoints on the right it's excluded
    # for all (N/2 choose K) permutations of just points on the left, and same on the right,
    # so below we subtract -2.0 * (N/2 choose K) / (N choose K) twice for that.
    # It's -2.0 times that because it's a count of moves to reach every square and get back
    # to where you started - so each square you move into and out of.
    for a in arr_all_vertex:
        # print("vertex", a.index)
        for e in a.adj_edge:
            ai_cntHereByDist = countCheckpoints(e.vertex_dest, a)
            # print(a.index, f"{ai_cntHereByDist=}")
            cntHere = 0
            for x in ai_cntHereByDist:
                cntHere += x
            assert cntHere == sum(ai_cntHereByDist)
            res -= 2.0 * c[cntHere][K] / c[totalCheckpoints][K]
            e.checkpointDistMap = ai_cntHereByDist

    # res is the expected number of moves to visit all the expected number of squares visited
    # What you have to do to start at a checkpoint and visit all the checkpoints and return to the start checkpoint
    # res = (2 * (number_of_squares_visited - 1))
    print(f"{res=}")    
    # So now we need to subtract off the length of the of the expected longest path between the K chosen checkpoints
    # Because we don't have to return to the start, so the start and end checkpoints will be the longest path
    # Odd diameter
    for a in arr_all_vertex:
        for ab in a.adj_edge:
            b = ab.vertex_dest
            if a.index > b.index:
                continue
            ba = None
            for ee in ab.vertex_dest.adj_edge:
                if ee.vertex_dest == a:
                    ba = ee
            countAb = 0
            countBa = 0
            for dist in range(totalVertices):
                savedCountab = countAb
                savedCountba = countBa
                countAb += ab.checkpointDistMap[dist]
                countBa += ba.checkpointDistMap[dist]
                cur = (c[countAb + countBa][K] - c[savedCountab + countBa][K] - 
                       c[countAb + savedCountba][K] + c[savedCountab + savedCountba][K]) / c[totalCheckpoints][K]
                res -= cur * (dist * 2 + 1)
    print(f"{res=}")
    # Even diameter
    for a in arr_all_vertex:
        for b in a.adj_edge:
            b.count = 0

        for dist in range(totalVertices):
            totalCount = 0
            totalSavedCount = 0
            for b in a.adj_edge:
                b.savedCount = b.count
                b.count += b.checkpointDistMap[dist]
                totalCount += b.count
                totalSavedCount += b.savedCount
            cur = 0.0
            if a.checkpoint > 0:
                totalCount += 1
            cur += c[totalCount][K]
            if a.checkpoint > 0:
                totalSavedCount += 1
            for b in a.adj_edge:
                cur -= c[totalSavedCount - b.savedCount + b.count][K]

            cur += c[totalSavedCount][K] * (len(a.adj_edge) - 1)
            cur /= c[totalCheckpoints][K]
            res -= cur * (2 * dist + 2)
    return res

field = [
 "*#..#",
 ".#*#.",
 "*...*"]
K = 2
print("Example1")
print("Example1 expected_length():", expected_length(field, K))
print("Example1 correct answer is: 3.8333333333333353")
print("\n")

field = [
 "*#..#",
 ".#*#.",
 "*...*"]
K = 4
print("Example2")
print("Example2 expected_length():", expected_length(field, K))
print("Example2 correct answer is: 8.0")
print("\n")

field = [
 "#.#**",
 "....#",
 "#*#**",
 "**#*#",
 "#..##",
 "*#..#",
 ".#.#.",
 "....*"]
K = 3
print("Example3")
print("Example3 expected_length():", expected_length(field, K))
print("Example3 correct answer is: 10.825000000000024")
print("\n")
field = [    	
 "###################",
 "#*###############*#",
 "#.....#######.....#",
 "#*###*.#.*.#.*###*#",
 "#*####*.*#*.*####*#",
 "#*#####*###*#####*#",
 "###################"]
K = 9
print("Example4")
print("Example4 expected_length():", expected_length(field, K))
print("Example4 correct answer is: 30.272233648704244")
print("\n")

field = [
 "**##*.**#..#.*...*#...*#..#.##..#..#.#*...#.##*##.", # 10  10
 ".#..###..#..#.#.##..#.#.*#.*..#..#.#*..##.#*...*..", # 5   15
 "..#.....###.#*.##..#.#.#*..#.#..#....#..#...#*####", # 3   18
 ".#.##*#.*#..#*#*.#.#...*.#.*#.#.##.#*.##.#.#..*...", # 8   26
 "..*.*#*.###.#..#.#..##.##.*#..#.....#.....#..#.#.#",
 ".#.##.#..##..*#..#.#...#*##*#*..#.#.#.#.##.##.#.#*",
 "..##....#..#.#*#...*.##...#.#.####...#.#*.....#...",
 ".#.*#.##.*#*.#*.#.#.#..#.#..#.#*#.###..##.##.#.##*",
 ".*.#*..*.#.#...#.*##.#.**.#.*...**..*#..#.#.#*.#..",
 ".#*.#*##....##.#.#*..*.###.#.##.##.#.#.#....#.#*.#",
 "*.#..#*#.#*#*....#.#.#..*#**...##.#.#.**#*##.*.#..",
 ".#*.##..##..##.#.#..#.#.###.###...#...#*#..##*#.#.",
 "#..#*.#..*.###..#.#...#.###.#.#*#.#.#**##.#...*.#*",
 "..#..#.#.##.#..#.**.##*#.#**.**..#.#..#...#.##*#..",
 ".#*#.#.*..#.*#...#.#...#...#.##.#..*#*.##*....###.",
 ".*.#.#.#.#*#..*##.**.##*##..#.*#.#*###..*.#.##.#..",
 ".#......#...#.#.*#.#.#..#..#.#*#....#*.#*#.*#..*.#",
 "#..####..#*#...#*.#..#.###...#.#.#.###*#..##*##.#.",
 ".#.*..#.#...#.#..#.##...#..#.#.#.#.###..##..*.*.*.",
 ".#.#.#.#..##.*..#.*.#.##.#..##*...#.#..#.#.##.#.##",
 ".#..#*.#.#..#.##..##..#.*..#.*#.#...##....#...###.",
 ".#.#.#.#*.#.#..#.#..#..#.#.*#...#.##...#.##.##.*..",
 ".#...#.#.##.#.#..*#.*#..###..#.#.#*###.##...#*.##.",
 ".#.##.*.......*.#.*#.#.#*###..*...*..#.*.##.#.#..#",
 "...###*####*#.#..##*...#..#..##.#.#.#..##*#*.*.*#.",
 "#.#.#....*#..#.#.#.#.##..#*.#...#..#.#*#...#.##.*.",
 "..*.#*##.#.#*#.###...#..##.#.#.#*###*#.*#.#.*###.#",
 "##*##..##...#.....##.#.#.**#..#*.....##.#..#*.#.*.",
 ".....#.*.##..##.##*.*#...#.#.#.##.#*#.**..#..#.#.#",
 "##.#.#*##.#.#.*.*.#.#*#.#.#....*...#*##*##.#....#.",
 "*.**#**....*..##.#*.*.**..##.###.##.....##...##.**",
 "#.####.##*#*##..#.*#*#.##*...#.##..#.##....#*..##.",
 "....#...##.#...#*.#..##.##.#*..*.#....##.#.*##...#",
 "#.#..*##*..#.#..#..#..#*....#.##..##.#*##.##.*##..",
 "..#.#*.*.##.#.#*#.#*##.###.##...#............#*.#.",
 "#.#.##.#....*....*..##..*#.#.#.###.#.#.#.###..#..#",
 ".#**..#*#.#*#*#.#.#...*##....##.#*..#..#*..*#..#..",
 "...#*#.....#..#.#..#*#.*##.#..#.#.##..#.*#*#.#...#",
 ".#*.###.#.#.#.#.*#*##.##..#.#*..#...#.#.#..#*.*#..",
 "#*.#.#.#..#..#..#....*#.*##..##.#.#..#...##.#.#..#",
 "*.#..#..#...#..##.#*#..#.#*#.#.#.###..#.#*...#.#..",
 "#...#.#...#.#.#..#.*.#*.....**.*..#*##.#*.##....##",
 "#*#....#*#..#.*.###*#..#*##.##.#.#...#.*.##.##.##.",
 "..##*##*..#*#.#..#*.*##*.##.#...#.#.#.#.#..*#.##..",
 "#...#*##.#*#**.##.*#.*.##..*.#*#**....#**##...*.*#",
 "*#.##......*#.##.#.#.##**.#.#.#.#.#.##..#...#*#*#*",
 "*....##.#.#..#.....#..##.#....*....#.#.##.#.#.##**",
 "#.##*#...#..#.#.##..#..##.##.##.##........##.#*#.#",
 "..#...#.#*#*..*#..*#.*#.#......##.#.#.#*#..#..****",
 ".###.#..#...#.#..#..#.#...#.#.#...**.#..*#*.*##*#."]
K = 150
print("Example5")
print("Example5 expected_length():", expected_length(field, K))
print("Example5 correct answer is: 1309.4951033725558")

'''
Problem Statement
Mrs. Qiu is planning to practice orienteering. The area where she'll practice is a rectangular field divided into unit squares. You are given its description as a String[] field. Each character in field is '.' (a period), '*' (an asterisk), or '#' (a number sign). Each '.' represents a passable square without a checkpoint, each '*' represents a passable square with a checkpoint, and each '#' represents an impassable obstacle. It is guaranteed that all passable squares (i.e., all '.'s and '*'s) form a 4-connected tree (see notes for formal definition). The number of checkpoints is at most 300. 

In order to practice, Mrs. Qiu chooses K of the checkpoints uniformly at random. Afterwards, she will find the shortest sequence of squares that passes through all chosen checkpoints. The sequence can start at any square, end at any square (possibly other than the starting one), and visit each square any number of times. Each pair of consecutive squares in the sequence must have a common side. The length of the sequence is the number of moves Mrs. Qiu will have to make. (So, for example, a sequence that consists of 7 squares has length 6.) 

You are given the String[] field and the int K. Return the expected length of Mrs. Qiu’s sequence.

Definition
Method signature:	def expected_length(field: List[str], K: int) -> float

Notes
-	A set S of squares is said to form a 4-connected tree if for any two squares A and B from S, there exists exactly one way to walk from A to B while visiting only the squares from S and not visiting the same square more than once. From a given square, it is possible to walk into any square that shares a common side with it.
Constraints
-	field will contain between 1 and 50 elements, inclusive.
-	Each element of field will contain between 1 and 50 characters, inclusive.
-	Each element of field will contain the same number of characters.
-	Each character in field will be '*', '.', or '#'.
-	'*' and '.' form a 4-connected tree.
-	K will be between 2 and 300, inclusive.
-	field will contain between K and 300 '*', inclusive.

Examples

Example #1:
field = [
 "*#..#",
 ".#*#.",
 "*...*"]
K = 2
Returns: 3.8333333333333353

Explanation:
Let (i,j) be the square represented by the j-th character of the i-th element of field (both numbers are 0-based). 

If she chooses (0,0) and (1,2), one of the optimal sequences is (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (1,2).
If she chooses (0,0) and (2,0), one of the optimal sequences is (0,0) -> (1,0) -> (2,0).
If she chooses (0,0) and (2,4), one of the optimal sequences is (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3) -> (2,4).
If she chooses (1,2) and (2,0), one of the optimal sequences is (1,2) -> (2,2) -> (2,1) -> (2,0).
If she chooses (1,2) and (2,4), one of the optimal sequences is (1,2) -> (2,2) -> (2,3) -> (2,4).
If she chooses (2,0) and (2,4), one of the optimal sequences is (2,0) -> (2,1) -> (2,2) -> (2,3) -> (2,4).
If she chooses (0,0), (1,2), and (2,4)
So the expected length of her sequences is:
  (5 + 2 + 6 + 3 + 3 + 4) / 6 = 23 / 6 = 3.8333333333333353


Example #2:
field = [
 "*#..#",
 ".#*#.",
 "*...*"]
K = 4
Returns: 8.0

Explanation:
Mrs. Qiu chooses all four checkpoints. One of the shortest sequences is (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (1,2) -> (2,2) -> (2,3) -> (2,4).


Example #3:
field = [
 "#.#**",
 "....#",
 "#*#**",
 "**#*#",
 "#..##",
 "*#..#",
 ".#.#.",
 "....*"]
K = 3
Returns: 10.825000000000024


Example #4:
field = [    	
 "###################",
 "#*###############*#",
 "#.....#######.....#",
 "#*###*.#.*.#.*###*#",
 "#*####*.*#*.*####*#",
 "#*#####*###*#####*#",
 "###################"]
K = 9
Returns: 30.272233648704244


Example #5:
field = [    	
 "**##*.**#..#.*...*#...*#..#.##..#..#.#*...#.##*##.",
 ".#..###..#..#.#.##..#.#.*#.*..#..#.#*..##.#*...*..",
 "..#.....###.#*.##..#.#.#*..#.#..#....#..#...#*####",
 ".#.##*#.*#..#*#*.#.#...*.#.*#.#.##.#*.##.#.#..*...",
 "..*.*#*.###.#..#.#..##.##.*#..#.....#.....#..#.#.#",
 ".#.##.#..##..*#..#.#...#*##*#*..#.#.#.#.##.##.#.#*",
 "..##....#..#.#*#...*.##...#.#.####...#.#*.....#...",
 ".#.*#.##.*#*.#*.#.#.#..#.#..#.#*#.###..##.##.#.##*",
 ".*.#*..*.#.#...#.*##.#.**.#.*...**..*#..#.#.#*.#..",
 ".#*.#*##....##.#.#*..*.###.#.##.##.#.#.#....#.#*.#",
 "*.#..#*#.#*#*....#.#.#..*#**...##.#.#.**#*##.*.#..",
 ".#*.##..##..##.#.#..#.#.###.###...#...#*#..##*#.#.",
 "#..#*.#..*.###..#.#...#.###.#.#*#.#.#**##.#...*.#*",
 "..#..#.#.##.#..#.**.##*#.#**.**..#.#..#...#.##*#..",
 ".#*#.#.*..#.*#...#.#...#...#.##.#..*#*.##*....###.",
 ".*.#.#.#.#*#..*##.**.##*##..#.*#.#*###..*.#.##.#..",
 ".#......#...#.#.*#.#.#..#..#.#*#....#*.#*#.*#..*.#",
 "#..####..#*#...#*.#..#.###...#.#.#.###*#..##*##.#.",
 ".#.*..#.#...#.#..#.##...#..#.#.#.#.###..##..*.*.*.",
 ".#.#.#.#..##.*..#.*.#.##.#..##*...#.#..#.#.##.#.##",
 ".#..#*.#.#..#.##..##..#.*..#.*#.#...##....#...###.",
 ".#.#.#.#*.#.#..#.#..#..#.#.*#...#.##...#.##.##.*..",
 ".#...#.#.##.#.#..*#.*#..###..#.#.#*###.##...#*.##.",
 ".#.##.*.......*.#.*#.#.#*###..*...*..#.*.##.#.#..#",
 "...###*####*#.#..##*...#..#..##.#.#.#..##*#*.*.*#.",
 "#.#.#....*#..#.#.#.#.##..#*.#...#..#.#*#...#.##.*.",
 "..*.#*##.#.#*#.###...#..##.#.#.#*###*#.*#.#.*###.#",
 "##*##..##...#.....##.#.#.**#..#*.....##.#..#*.#.*.",
 ".....#.*.##..##.##*.*#...#.#.#.##.#*#.**..#..#.#.#",
 "##.#.#*##.#.#.*.*.#.#*#.#.#....*...#*##*##.#....#.",
 "*.**#**....*..##.#*.*.**..##.###.##.....##...##.**",
 "#.####.##*#*##..#.*#*#.##*...#.##..#.##....#*..##.",
 "....#...##.#...#*.#..##.##.#*..*.#....##.#.*##...#",
 "#.#..*##*..#.#..#..#..#*....#.##..##.#*##.##.*##..",
 "..#.#*.*.##.#.#*#.#*##.###.##...#............#*.#.",
 "#.#.##.#....*....*..##..*#.#.#.###.#.#.#.###..#..#",
 ".#**..#*#.#*#*#.#.#...*##....##.#*..#..#*..*#..#..",
 "...#*#.....#..#.#..#*#.*##.#..#.#.##..#.*#*#.#...#",
 ".#*.###.#.#.#.#.*#*##.##..#.#*..#...#.#.#..#*.*#..",
 "#*.#.#.#..#..#..#....*#.*##..##.#.#..#...##.#.#..#",
 "*.#..#..#...#..##.#*#..#.#*#.#.#.###..#.#*...#.#..",
 "#...#.#...#.#.#..#.*.#*.....**.*..#*##.#*.##....##",
 "#*#....#*#..#.*.###*#..#*##.##.#.#...#.*.##.##.##.",
 "..##*##*..#*#.#..#*.*##*.##.#...#.#.#.#.#..*#.##..",
 "#...#*##.#*#**.##.*#.*.##..*.#*#**....#**##...*.*#",
 "*#.##......*#.##.#.#.##**.#.#.#.#.#.##..#...#*#*#*",
 "*....##.#.#..#.....#..##.#....*....#.#.##.#.#.##**",
 "#.##*#...#..#.#.##..#..##.##.##.##........##.#*#.#",
 "..#...#.#*#*..*#..*#.*#.#......##.#.#.#*#..#..****",
 ".###.#..#...#.#..#..#.#...#.#.#...**.#..*#*.*##*#."]
K = 150
Returns: 1309.4951033725558
'''