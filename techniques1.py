# GREEDY STYLE
def coinChange(totalNumber, coins):
    N = totalNumber
    coins.sort()
    index = len(coins)-1
    while True:
        coinValue = coins[index]
        if N >= coinValue:
            print(coinValue)
            N = N - coinValue
        if N < coinValue:
            index -= 1
        
        if N == 0:
            break

coins = [1,2,5,20,50,100]
coinChange(230, coins)


def maximumUnits(boxTypes, truckSize):

        """
        :type boxTypes: List[List[int]]
        :type truckSize: int
        :rtype: int
        """
        sum = 0
        boxTypes = sorted(boxTypes , key = lambda i:i[1])

        while truckSize != 0:
            if len(boxTypes)==0:
                break

            if truckSize <=0:
                break

            if boxTypes[-1][0] < truckSize:
                sum = sum + boxTypes[-1][0]*boxTypes[-1][1]
                truckSize -= boxTypes[-1][0] 
                boxTypes.pop()
                continue

            if truckSize > 0  and boxTypes[-1][0] >= truckSize:
                sum = sum + truckSize*boxTypes[-1][1]
                truckSize = 0

        return sum  

boxTypes = [[1,3],[5,5],[2,5],[4,2],[4,1],[3,1],[2,2],[1,3],[2,5],[3,2]]
truckSize = 35
print(maximumUnits(boxTypes,truckSize))

################################################################################################################################
# Divide and Conquer Approach
# Question-1 : Combination Sum when number of elements in array is very low and the elments are known
def numberOfFactor(n):
    if n in (0,1,2):
        return 1
    elif n == 3:
        return 2
    else:
        sp1 = numberOfFactor(n-1)
        sp2 = numberOfFactor(n-3)
        sp3 = numberOfFactor(n-4)
        return sp1 + sp2 + sp3
    
print(numberOfFactor(5))

def lol_test(arr):
    arr = [[arr[i],i] for i in range(len(arr))]
    arr = sorted(arr , key= lambda a:a[0])
    return arr


# Question-2 : House Robber 1
def rob(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return nums[0]
        n = len(nums)
        dp = [0] * n
        
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, n):
            dp[i] = max(dp[i-1], nums[i] + dp[i-2]) # at each cell every alternate cells are added
        
        return dp[-1]

arr = [1,1,1]

print(lol_test(arr))

print(rob(arr))

################################################################################################################################
# Morris Traversal
# Q:- Tree to Linked List

def flatten(root):
        curr = root
        while curr:
            if curr.left:
                runner = curr.left
                while runner.right: runner = runner.right
                runner.right, curr.right, curr.left = curr.right, curr.left, None
            curr = curr.right
################################################################################################################################

# Q:- Divide and conquer

def numberOfPaths(twoDArray, row, col, cost):
    if cost < 0:
        return 0
    elif row == 0 and col == 0:
        if twoDArray[0][0] - cost == 0:
            return 1
        else:
            return 0
    elif row == 0:
        return numberOfPaths(twoDArray, 0, col-1, cost - twoDArray[row][col] )
    elif col == 0:
        return numberOfPaths(twoDArray, row-1, 0, cost - twoDArray[row][col] )
    else:
        op1 = numberOfPaths(twoDArray, row -1, col, cost - twoDArray[row][col] )
        op2 = numberOfPaths(twoDArray, row, col-1, cost - twoDArray[row][col] )
        return op1 + op2


TwoDList = [[4,7,1,6],
           [5,7,3,9],
           [3,2,1,2],
           [7,1,6,3]
           ]

print(numberOfPaths(TwoDList, 3,3, 25))

def longest_palindrome(string , start , last):
    if start>last:     
        return 0
    elif start == last:
        return 1
    elif string[start] == string[last]:
        return 2 + longest_palindrome(string , start+1 , last-1)
    else:
        op1 = longest_palindrome(string , start+1 , last)
        op2 = longest_palindrome(string , start , last-1)
        return max(op1,op2) 

string = "ELRMENMET"    
print('palindrome')
print(longest_palindrome(string , 0 , len(string)-1))


def canCompleteCircuit(gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        def drt(start , gas , cost):
            tank = gas[start] - cost[start]
            temp_index = start
            count = 0
            while tank >= 0:
                if count == len(gas) - 1:
                    return start
                temp_index += 1
                if temp_index > len(gas)-1 and count != len(gas) - 1:
                    temp_index = 0
                tank +=  gas[temp_index] - cost[temp_index]
                count += 1
            return -1
        
        for i in range(len(gas)):
            if drt(i , gas , cost) != -1:
                return drt(i , gas , cost)
            else:
                continue
        return -1

gas = [2,3,4]
cost = [3,4,3]
print(canCompleteCircuit(gas , cost))

################################################################################################################################
# Union Find (Very Important)
def minSwapsCouples(row):
        """
        :type row: List[int]
        :rtype: int
        """
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x,y):
            xRep = find(x)
            yRep = find(y)
            if find(x) != find(y):
                parent[xRep] = yRep

        n = len(row)//2
        parent = [i for i in range(n)]

        for i in range(0,len(row),2):
            union(row[i]//2 , row[i+1]//2)

        count = sum([1 for i,x in enumerate(parent) if i == find(x)])
        return n - count

################################################################################################################################
# Mathematical approach
def uniquePaths(m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        res = [[1]*n]*m
        for row in range(1,m):
            for col in range(1,n):
                res[row][col] = res[row-1][col] + res[row][col-1]
            
        return res[-1][-1]

print(uniquePaths(3,2))
################################################################################################################################

# DYNAMIC PROGRAMMING
def string_conversion(s1,s2,ind1,ind2,temp_dict):
    if ind1 == len(s1):
        return len(s2) - ind1
    if ind2 == len(s2):
        return len(s1) - ind2
    if s1[ind1] == s2[ind2]:
        return 1 + string_conversion(s1,s2,ind1+1,ind2+1,temp_dict)
    else:
        key = s1[ind1]+s2[ind2]
        if key not in temp_dict:
                delete = string_conversion(s1,s2,ind1,ind2+1,temp_dict)+1
                insert = string_conversion(s1,s2,ind1+1,ind2,temp_dict)+1
                replace = string_conversion(s1,s2,ind1+1,ind2+1,temp_dict)+1
                temp_dict[key] = min(delete,insert,replace)
        return temp_dict[key]
print('lllll')
print(string_conversion("llll",'aarch',0,0,dict()))

################################################################################################################################
import heapq
# Shortest Path
class Solution(object):
    def findCheapestPrice(self, n, flights, src, dst, k):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type k: int
        :rtype: int
        """
        adj = [[] for i in range(n)]
        for i, j, l in flights:
            adj[i].append([j, l])
        print(adj)

        dis = [float("inf") for _ in range(n)]
        dis[src] = 0
        heap = [(0, 0, src)]

        while heap:
            stops, price, city = heapq.heappop(heap)
            for c in adj[city]:
                if c[0] == dst:
                    if stops <= k and (price + c[1]) < dis[dst]:
                        dis[dst] = price + c[1]
                else:
                    if stops < k and (price + c[1]) < dis[c[0]]:
                        dis[c[0]] = price + c[1]
                        heapq.heappush(heap, (stops + 1, price + c[1], c[0]))
                        
        if dis[dst] == float("inf"):
            return -1
        return dis[dst]

flights = [[0,1,100],[1,2,100],[0,2,500]]    
print(Solution().findCheapestPrice(len(flights) , flights , 0 , 2 , 3))

def missingNumber(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        for i in range(0,len(nums)):
            if i != nums[i]:
                return i
        return len(nums) 

print(missingNumber([9,1,2,3,0,5,6,7,8]))


def productExceptSelf(nums):
        total_product = 1
        zero_count = 0
        
        for num in nums:
            if num != 0:
                total_product *= num
            else:
                zero_count += 1
                
        result = []
        
        if zero_count > 1:
            return [0] * len(nums)
        
        for num in nums:
            if num != 0:
                if zero_count == 1:
                    result.append(0)
                else:
                    result.append(total_product // num)
            else:
                result.append(total_product)
        
        return result 


def maxProduct(nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        result = maxi = mini = nums[0]

        for i in range(1,len(nums)):
            
            maxi = max(nums[i] , maxi*nums[i] , mini*nums[i])
            mini = min(nums[i] , maxi*nums[i] , mini*nums[i])
            result = max(result,maxi)

        return result

print(maxProduct([-4,-3,-2]))

#############################################################################################################################
# DP bottom-up tabulation

def longestCommonSubsequence( S1, S2):
        """
        :type text1: str
        :type text2: str
        :rtype: int
        """
        m = len(S1)
        n = len(S2)

        # Initializing a matrix of size (m+1)*(n+1)
        dp = [[0] * (n + 1) for x in range(m + 1)]

        # Building dp[m+1][n+1] in bottom-up fashion
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if S1[i - 1] == S2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j],
                                dp[i][j - 1])

        # dp[m][n] contains length of LCS for S1[0..m-1]
        # and S2[0..n-1]
        return dp[m][n]

################################################################################################################################
# Bactracking + Trie
'''from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(list)
        self.isWord = False

class Trie:
    def __init__(self):
       self.root = TrieNode()

    def insert(self,word):
        curr = self.root
        for char in word:
            curr_node = curr.children.get(char)
            if curr_node is None:
                node = TrieNode()
                curr.children[char] = node
            curr = curr_node
        curr.isWord = True

    def search(self, word):
        curr = self.root
        for w in word:
            node = curr.children.get(w)
            if not node:
                return False
            curr = node
        return curr.isWord
        
        
class Solution(object):
    def __init__(self):
        self.res = []

    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        t = Trie()
        for word in words:
            t.insert(word)

        def backtrack(node,i,j,path):
            if node.isWord:
                self.res.append(path)
                node.isWord = False

            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) :
                return 

            temp = board[i][j]
            node = node.children.get(temp)
            
            if node is None:
                return
            
            board[i][j] = ''

            backtrack(node,i+1,j,path+temp)
            backtrack(node,i,j+1,path+temp)
            backtrack(node,i,j-1,path+temp)
            backtrack(node,i-1,j,path+temp)

            board[i][j] = temp

        backtrack(t.root , 0 , 0 , '')

        return self.res
    
board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
words = ["oath","pea","eat","rain"]


s = Solution()
print(s.findWords(board , words))'''

#################################################################################################################
# Moore Voting Algorithm
# Q:- Majority Element (leetcode-169)
def majorityele(nums):
        candidate1, candidate2 = None, None
        count1, count2 = 0, 0
      
        # Perform Boyer-Moore Majority Vote algorithm
        for num in nums:
            # If the current element is equal to one of the potential candidates,
            # increment their respective counters.
            if num == candidate1:
                count1 += 1
            elif num == candidate2:
                count2 += 1
            # If one of the counters becomes zero, replace the candidate with
            # the current element and reset the counter to one.
            elif count1 == 0:
                candidate1, count1 = num, 1
            elif count2 == 0:
                candidate2, count2 = num, 1
            # If the current element is not equal to any candidate and both
            # counters are non-zero, decrement both counters.
            else:
                count1 -= 1
                count2 -= 1
      
        # Check whether the candidates are legitimate majority elements.
        # A majority element must appear more than len(nums) / 3 times.
        return [m for m in (candidate1, candidate2) if nums.count(m) > len(nums) // 3]

################################################################################################################################
# IPO (Slightly similar to gas station prob)

def findMaximizedCapital(k, w, profits, capital):
        vp = list(zip(capital,profits))

        vp.sort()

        queue = []

        profit = w
        j = 0

        for i in range(k):
            while j<len(profits) and vp[j][0] <= w:
                heapq.heappush(queue , -vp[j][1])
                j += 1
            if queue:
                w -= heapq.heappop(queue)
            else:
                break

        return vp

       #***********************************************
       # MY SOLUTION (testcase passed --> 25/40)
'''     vp = []

        for i in range(len(capital)):
            vp.append((capital[i],profits[i]))
        
        hash = {key : False for key in vp}

        vp = sorted(vp , key = lambda a: a[0]-a[1])
        
        for j in range(k):
            temp = value = -1
            for i in range(len(capital)):
                if vp[i][0] <= w and not hash[vp[i]] and vp[i][1] > value:
                    temp = vp[i]
                    value = vp[i][1]

            if temp == -1:
                break
            w += value
            hash[temp] = True

        return w'''

########################################################################################################################
print(findMaximizedCapital(k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]))

from collections import defaultdict
routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]]

adj_list = defaultdict(set)

for group, route in enumerate(routes):
            for stop in route:
                adj_list[stop].add(group)
print(adj_list)

t = '1'*(4)
print(t)