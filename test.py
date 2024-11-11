#
# Complete the 'palindromeIndex' function below.
#
# The function is expected to return an INTEGER.
# The function accepts STRING s as parameter.
#


for lo in range(0, 1000):
    for hi in range(0, 1000):
        if lo + (hi - lo) // 2 !=  (lo + hi) // 2:
            print(lo + (hi - lo) // 2, (lo + hi) // 2)

for i in range(1, 0):
    print(i)

exit()    


def is_palindrome(s):
    # print("is_palindrome", s)
    if len(s) < 2:
        return True
    start = 0
    end = len(s) - 1
    
    while start < end:
        if s[start] != s[end]:
            return False
        start += 1
        end -= 1
    
    return True

def palindromeIndex(s):
    # Write your code here
    # print("palindromIndex", s)

    if is_palindrome(s):
        return -1
    
    start = 0    
    end = len(s) - 1
    
    while start < end:
        if s[start] != s[end]:
            # Remove the start, keep the end
            if is_palindrome(s[start + 1 : end + 1]):
                return start
            # Remove the end, keep the start
            if is_palindrome(s[start : end]):
                return end
            return -1
        start += 1
        end -= 1
    
    # for i in range(len(s)):
    #    if is_palindrome(s[:i] + s[i+1:]):
    #        return i
    return -1

assert palindromeIndex('bacadb') == 4 #  s = 'aaab' (first query)
assert palindromeIndex('baccdab') == 4 #  s = 'aaab' (first query)
assert palindromeIndex('bacabd') == 5 #  s = 'aaab' (first query)
assert palindromeIndex('baaa') == 0 #  s = 'aaab' (first query)
assert palindromeIndex('baaaa') == 0 #  s = 'aaab' (first query)
assert palindromeIndex('abaa') == 1 #  s = 'aaab' (first query)
assert palindromeIndex('abaaa') == 1 #  s = 'aaab' (first query)
assert palindromeIndex('dabd') == 1 #  s = 'aaab' (first query)
assert palindromeIndex('dabaad') == 2 #  s = 'aaab' (first query)
assert palindromeIndex('aaab') == 3 #  s = 'aaab' (first query)
assert palindromeIndex('aaaab') == 4 #  s = 'aaab' (first query)
assert palindromeIndex('baa') == 0 # s = 'baa' (second query)
assert palindromeIndex('bcbc') == 0 # 
assert palindromeIndex('') == -1 # 
assert palindromeIndex('a') == -1 # 
assert palindromeIndex('aa') == -1 # 
assert palindromeIndex('aba') == -1 # 
assert palindromeIndex('aaa') == -1 # 
assert palindromeIndex('aaaa') == -1 # 
assert palindromeIndex('abba') == -1 # 
assert palindromeIndex('baaab') == -1 # 
assert palindromeIndex('ababa') == -1 # 
assert palindromeIndex('aabaa') == -1 # 
assert palindromeIndex('bababab') == -1 # 
assert palindromeIndex('abc') == -1 
assert palindromeIndex('abcd') == -1 
assert palindromeIndex('abcde') == -1 
assert palindromeIndex('aabcdeaa') == -1

import math
import os
import random
import re
import sys
import itertools

def AddBlock(rowPermutations, rowSoFar, nLeft):
    if nLeft == 0:
        rowPermutations.append(rowSoFar.copy())
        return
    if nLeft >= 1:
        rowSoFar.append(1)
        AddBlock(rowPermutations, rowSoFar, nLeft - 1)
        rowSoFar.pop()
    if nLeft >= 2:
        rowSoFar.append(2)
        AddBlock(rowPermutations, rowSoFar, nLeft - 2)
        rowSoFar.pop()
    if nLeft >= 3:
        rowSoFar.append(3)
        AddBlock(rowPermutations, rowSoFar, nLeft - 3)
        rowSoFar.pop()
    if nLeft >= 4:
        rowSoFar.append(4)
        AddBlock(rowPermutations, rowSoFar, nLeft - 4)
        rowSoFar.pop()
    return
        
def legoBlocks(n, m):
    # Write your code here
    # Compute how many layouts are possible in a row
    # Store that as a list of lists of block types (1,2,3,4)

    rowPermutations = []
    AddBlock(rowPermutations, [], m)
    print("rowPermutations", rowPermutations)

    # So you could have cRowPermutations in a row, raised to the N power
    cRowPermutations = len(rowPermutations) ** n  # If no bad layouts
    print("cRowPermutations with bad layouts", cRowPermutations)
    
    # But you have to subtract out the count of permutations that all end on the same crack
    # So iterate through all permutations with replacement
    # Sum the indexes across the rows so each index is the index where a block ends
    # If all the rows end at any of the same spots - it's a bad layout
    
    for row_index in range(len(rowPermutations)):
        for col_index in range(1, len(rowPermutations[row_index])):
            rowPermutations[row_index][col_index] += rowPermutations[row_index][col_index - 1]
    print("rowPermutations End", rowPermutations)
                
    cRowPermutations = 0
    cAll = 0
    
    index_row = [i for i in range(len(rowPermutations))]
    print("index_row", index_row)
                    
    for layout in itertools.product(index_row, repeat=n):
        cAll += 1
        # if n <= 3:
        #     print("layout", layout)
        
        for col_index in range(1, m): # All possible endings
            bAllEnd = True # Do all rows end there?
            for row_index in range(len(layout)):
                if col_index not in rowPermutations[layout[row_index]]:
                    bAllEnd = False
                    break
            
            if bAllEnd == True:  # Invalid layout
                break
                
        if bAllEnd == False:
            cRowPermutations += 1 # Another valid layout

    print("cRowPermutations no bad layouts, cAll", cRowPermutations, cAll)
    return cRowPermutations

def legoBlocks(h, w):
    modulo = 10**9 + 7
    rowPermutations = [0] * (w + 1)
    if w >= 1:
        rowPermutations[1] = 1
    if w >= 2:
        rowPermutations[2] = 2
    if w >= 3:
        rowPermutations[3] = 4
    if w >= 4:
        rowPermutations[4] = 8
    if w >= 5:
        for i in range(5, w + 1):
            rowPermutations[i] = (rowPermutations[i - 1] + rowPermutations[i - 2] + rowPermutations[i - 3] + rowPermutations[i - 4]) % modulo
            # rowPermutations[i] = rowPermutations[i] % modulo
    # print("rowPermutations", rowPermutations)

    # Compute how many unique layouts are possible if you stack the rows vertically
    total = rowPermutations.copy()
    for _ in range(2, h+1):
        for i in range(1, w + 1):        
            total[i] = (rowPermutations[i] * total[i]) % modulo
            # total[i] = total[i] % modulo

    # print("total", total)

    solid = [0] * (w + 1)
    solid[1] = 1
    for ww in range(2, w+1):
        unsolid_sum = 0
        for i in range(1, ww):
            unsolid_sum += ((solid[i] * total[ww - i]) % modulo)
            unsolid_sum = unsolid_sum % modulo
        solid[ww] = total[ww] - unsolid_sum
    
    # print("solid", solid)
    return solid[w] % modulo

input1 = """694 335
440 335
422 160
986 958
355 762
763 973
542 717
853 851
663 483
400 218
155 174
16 507
852 365
791 264
492 173
38 538
860 829
872 281
988 857
591 342
971 353
666 512
70 518
362 84
352 113
301 507
639 668
365 490
33 155
105 876
680 142
413 539
970 637
171 957
845 761
650 815
466 315
327 887
184 40
970 536
153 622
394 791
290 110
632 674
265 736
549 296
878 314
834 199
950 356
156 794
469 157
961 934
824 287
172 359
678 141
246 182
762 991
324 51
101 955
76 365
43 625
660 920
290 845
470 239
552 977
384 20
134 344
305 957
982 476
667 12
968 913
193 730
903 869
132 3
175 208
719 217
184 378
488 473
574 958
63 126
934 798
497 419
142 154
727 475
981 394
486 949
306 31
30 560
899 161
563 425
720 281
642 903
11 481
727 584
790 141
709 724
939 558
494 432
711 221
906 691"""

output1 = """30314890
229443565
439007948
971197618
241895716
717284387
588839400
303101413
402874978
576792539
926344476
204881400
277218231
339139226
376372247
281276277
791892311
380042137
615792624
315052702
607839422
901426096
246736770
675854429
872969237
472855842
624000354
460948269
162162397
739668480
363844227
443651529
334933457
862810012
5291493
538460906
138299373
607380820
185418148
406901127
266627027
596082696
545249439
881176334
16099769
12044740
503777963
740364286
989281856
848495993
264197587
696083823
764168155
86177283
402777142
502979956
825475912
962622463
404162231
663308276
772225304
291809375
938079506
323688054
210850424
782623977
795866825
130345948
201736930
129707397
146767052
781930105
630267421
20153875
904043237
769695880
724201492
417665544
571755430
536877641
486809984
486026753
855492421
781360655
928457351
923928323
791710168
717025438
938126516
715373870
831626158
876763899
923441705
895441933
861870774
767860405
602083336
93543650
851574896
114973689"""

if False:
    
    input = '2 2\n3 2\n2 3\n4 4\n4 5\n4 6\n4 7\n5 4\n6 4\n7 4\n'
    output = '3\n7\n9\n3375\n35714\n447902\n5562914\n29791\n250047\n2048383\n'

    input_list = input1.rstrip().split('\n')
    output_list = output1.rstrip().split('\n')

    for t_itr in range(len(input_list)):
        first_multiple_input = input_list[t_itr].rstrip().split()

        n = int(first_multiple_input[0])

        m = int(first_multiple_input[1])
        # print("n, m", n, m)

        result = legoBlocks(n, m)
        if result != int(output_list[t_itr]):
            print(t_itr, "Got:", str(result), "Expected:", output_list[t_itr])

def mergeLists(head1, head2):
    head_r = None
    if head1 != None:
        if head2 != None:  # head1 and head2 both exist
            if head1.data < head2.data:  # head1 less than head2
                head_r = head1
                head1 = head1.next
            else:  # head2 less than head1
                head_r = head2
                head2 = head2.next
        else: # Just head1, no head2
            head_r = head1
            head1 = head1.next
    elif head2 != None: # No head1, just head2
        head_r = head2
        head2 = head2.next
    # else:
        # head_r = None # head1 and head2 are None
        
    head_cur = head_r
    while head1 != None or head2 != None:
        if head1 != None:
            if head2 != None:  # head1 and head2 both exist
                if head1.data < head2.data:  # head1 less than head2
                    head_cur.next = head1
                    head_cur = head1
                    head1 = head1.next
                else:  # head2 less than head1
                    head_cur.next = head2
                    head_cur = head2
                    head2 = head2.next
            else: # Just head1, no head2
                head_cur.next = head1
                head_cur = head1
                head1 = head1.next
        else: # elif head2 != None: # No head1, just head2
            head_cur.next = head2
            head_cur = head2
            head2 = head2.next
        # else:
            # head_r = None # head1 and head2 are None
        
    return head_r

#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'pairs' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER k
#  2. INTEGER_ARRAY arr
#

def pairs_n2(k, arr):  # order N^2 - too slow to pass tests
    # Write your code here
    c_return = 0
    k = abs(k)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if abs(arr[i] - arr[j]) == k:
                c_return += 1
                
    return c_return
    
def pairs(k, arr):  # order N^2 still worst case
    # Write your code here
    c_return = 0
    k = abs(k)
    arr.sort()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if abs(arr[i] - arr[j]) >= k:
                if abs(arr[i] - arr[j]) == k:
                    c_return += 1
                break
  
    return c_return
    
# Could do a N*LOG(N) - sort the list - then run through the list 
# and search for (value+k) in sorted list

def logfind(arr, iMin, iMax, value):    # iMin and iMax are inclusive
    while iMin <= iMax:
        iNew = (iMin + iMax) // 2
        if arr[iNew] == value:
            return iNew
        elif arr[iNew] < value:
            iMin = iNew + 1
        else:
            iMax = iNew - 1
    return -1

def pairs(k, arr):  # order N*LOG(N) worst case
    # Write your code here
    c_return = 0
    k = abs(k)
    arr.sort()
    for i in range(len(arr)):
        iFound = logfind(arr, i + 1, min(i + k, len(arr) - 1), arr[i] + k)
        if iFound != -1:
            c_return += 1
  
    return c_return

# 7 2
# 1 3 5 8 6 4 2
# 5

if __name__ == '__main__':
    # fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = "7 2".split()

    n = int(first_multiple_input[0])

    k = int(first_multiple_input[1])

    arr = list(map(int, "1 3 5 8 6 4 2".rstrip().split()))

    result = pairs(k, arr)

    print(str(result) + '\n')

    # fptr.close()

'''
5 2
1 5 3 4 2
3

10 1
363374326 364147530 61825163 1073065718 1281246024 1399469912 428047635 491595254 879792181 1069262793
0

7 2
1 3 5 8 6 4 2
5
'''