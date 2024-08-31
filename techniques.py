#  2-pointer
def longestValidParentheses(s):
        if len(s)==0 or len(s)==1:
            return 0
       
        right = len(s)-1
        left = 0
        while left<=right:
            if s[right] == ')' and s[left] =='(':
                return right-left
            if left == right:
                return 0
            if s[right]!=')':
                right -= 1
            if left == right:
                return 0
            if s[left]!='(':
                left += 1
            if left == right:
                return 0

s = '()))'
print(longestValidParentheses(s))

# four sum problem (two sum approach) using 2-pointers
def fourSum(nums, target):
        nums.sort()
        res = []

        for i in range(len(nums) - 3):
            # skip duplicate starting values
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            for j in range(i+1, len(nums) - 2):
                # skip duplicate starting values
                if j > i + 1 and nums[j] == nums[j-1]:
                    continue
                left, right = j + 1, len(nums) - 1

                while left < right:
                    four_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if four_sum == target:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        right -= 1

                        while left < right and nums[left] == nums[left - 1]:
                            left += 1
                        while left < right and nums[right] == nums[right + 1]:
                            right -= 1
                    elif four_sum < target:
                        left += 1
                    else:
                        right -= 1
        return res

################################################################################################################################
#Breaking problem to small parts
board = [["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
class Solution(object):
    def __init__(self, board):
        self.board = board
        """
        :type board: List[List[str]]
        :rtype: bool
        """
    def has_duplicate(self,lst):
        seen = set()
        for num in lst:
            if num!=".":
                if num in seen:
                    return True
                seen.add(num)
        return False

    def checkRow(self,row):
        return self.has_duplicate(self.board[row])

    def checkColumn(self,col):
        column = [self.board[i][col] for i in range(len(self.board))]
        return self.has_duplicate(column)

    def checkGrid(self,row,col):
        subgrid = [self.board[i][j]for i in range(row,row+3) for j in range (col,col+3)]
        return self.has_duplicate(subgrid)
    
    def helper(self):
        for i in range(9):
            if self.checkRow(i) or self.checkColumn(i):
                return False

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                if not self.checkGrid(i, j):
                    return False

        return True
    
s = Solution(board)
print(s.helper())

################################################################################################################################
#Backtracking (with,recursion)

def combinationSum(candidates, target):
        ans = []
        def combinationSumHelper(candidates,curr_index,curr_sum,curr_combi,target):
            if curr_sum == target:
                ans.append(curr_combi)
                return
            if curr_sum>target:
                return
            for i in range(len(candidates)):
                combinationSumHelper(candidates , i ,curr_sum + candidates[i] , curr_combi+[candidates[i]] , target)

        combinationSumHelper(candidates,0,0,[],target)
        return ans
lst = [[2,3,6,7],[5,4,6,7],[0,1,2,3]]
candidates = [2,3,6,7]
target = 7
print(candidates)
print(combinationSum(candidates,target))

################################################################################################################################
#top K elements (Binary Heap)
#Q:- print top K largest elements

import heapq        #Binary heap package(min heap)
arr = [1, 23, 12, 9, 30, 2, 50]
K = 3
def top_K_smallest(arr,k):
    temp = []
    for ele in arr:
        heapq.heappush(temp,ele)
        if len(temp)>k:
            heapq.heappop(temp)
    for _ in range(k):
        print(heapq.heappop(temp))

print(top_K_smallest(arr,K))

################################################################################################################################
#   Hashing Technique
#   Refer 'Trie' section

################################################################################################################################
# Sliding window
def lengthOfLongestSubstring(string):
    last_idx = {}
    max_len = 0

    # starting index of current 
    # window to calculate max_len
    start_idx = 0

    for i in range(0, len(string)):
      
        # Find the last index of str[i]
        # Update start_idx (starting index of current window)
        # as maximum of current value of start_idx and last
        # index plus 1
        if string[i] in last_idx:
            start_idx = max(start_idx, last_idx[string[i]] + 1)

        # Update result if we get a larger window
        max_len = max(max_len, i-start_idx + 1)

        # Update last index of current char.
        last_idx[string[i]] = i

    return max_len
print(lengthOfLongestSubstring('aabccdefghhhj'))

###########################################################################################################################
# Optimized Binary Search
# Question : smallest pair distance
# Method : Similar to binary search and mergeSort combined
# i.e. makes a pair of every consecutive element and then logic is applied

import heapq

class Solution(object):
    def __init__(self):
        self.ans = {}

    def isSmall(self,nums):
        left = 0
        right = len(nums) - 1 
        mid = (left+right)//2
        if len(nums) == 2:
            return [(nums[0],nums[1]) , abs(nums[0] - nums[1])]
        
        pair_left = self.isSmall(nums[0:mid+1])
        pair_right = self.isSmall(nums[mid:])

        self.ans[pair_left[0]] = pair_left[1]
        self.ans[pair_right[0]] = pair_right[1]
        
        if pair_left[1] > pair_right[1]:
            return pair_right
        else:
            return pair_left

    def smallestDistancePair(self, nums):
        nums.sort()
        return self.isSmall(nums)

nums = [1,3,4,6,8,10,11,11]
s = Solution()
print(s.smallestDistancePair(nums))
print(s.ans)
print(list(s.ans))

# Question : 2

def searchInsert(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if target > nums[-1]:
            return len(nums)
        elif target < nums[0]:
            return 0
        else:
            left = 0
            right = len(nums)-1
            while left<=right:
                m = (left+right)//2
                if nums[m] == target:
                    return m
                elif nums[m] > target:
                    right = m-1
                if nums[m] < target:
                    left = m+1
            if nums[left] > target:
                return left
            if nums[left] < target:
                return left+1
nums = [1,3,5,6,7,10,12]
target = 11
print(searchInsert(nums,target))


# Important
class Solution(object):
    def kSmallestPairs(self, nums1, nums2, k):
        queue = []
        def push(i, j):
            if i < len(nums1) and j < len(nums2):
                heapq.heappush(queue, [nums1[i] + nums2[j], i, j])
        push(0, 0)
        pairs = []
        while queue and len(pairs) < k:
            _, i, j = heapq.heappop(queue)
            pairs.append([nums1[i], nums2[j]])
            push(i, j + 1)
            if j == 0:
                push(i + 1, 0)
        return pairs
    
def twoSum(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        numMap = {}
        n = len(nums)

        # Build the hash table
        for i in range(n):
            numMap[nums[i]] = i

        nums.sort()
        left = 0
        right = len(nums)-1
        while left<right:
            sumo = nums[left] + nums[right]
            if sumo == target:
                return[numMap[nums[left]],numMap[nums[right]]]
            elif sumo > target:
                right -= 1
            else:
                left += 1
        return False

nums = [2,7,11,15]
target = 9
print(twoSum(nums,target))

words = ["oath","pea","eat","rain"]
words.sort(key = lambda a:len(a))
print(words)

###############################################################################################################################
# Binary search method
def median(nums1 , nums2):
    n = len(nums1)
    m = len(nums2)

    if n > m:
        return median(nums2 , nums1)
    left = 0
    right = n

    while left <= right:
        partitionA = (left+right)//2
        partitionB = (m+n+1)//2 - partitionA
        
        maxleftA = nums1[partitionA-1] if partitionA != 0 else float("-inf")  
        minrightA = nums1[partitionA] if partitionA != n else float("inf")
        maxleftB = nums2[partitionB-1] if partitionB != 0 else float("-inf")
        minrightB = nums2[partitionB] if partitionB != m else float("inf")

        if maxleftA <= minrightB and maxleftB <= minrightB:
            if (m+n)%2==0:
                return (max(maxleftA,maxleftB) + min(minrightA,minrightB))/2
            else:        
                return max(maxleftA,maxleftB)
        elif maxleftA > minrightB:
            right = partitionA-1
        else:
            left = partitionA + 1

print(median([1,2],[3,4]))

