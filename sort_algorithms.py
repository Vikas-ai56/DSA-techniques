#      Algorithm          TC         SC          Stable
#    bubble_sort        O(n^2)      O(1)           Y
#    selection_sort     O(n^2)      O(1)           N  
#    insertion_sort     O(n^2)      O(1)           Y
#    bucket_sort        O(NlogN)    O(n)           Y
#    merge_sort         O(NlogN)    O(n)           Y
#    quicksort          O(NlogN)    O(n)           N
#    heapSort           O(NlogN)    O(1)           N

# Note :- merge_sort , quicksort  uses greedy algorithm and divide and Conquer Algorithm
#         because it only care about the local solution for each subarray
# Note :- selection_sort , insertion_sort uses greedy algorithm
#         because it only care about the local solution after each iteration

import math as m
lst = [9,1,2,3,6,88,7,5]
print(lst)

def bubble_sort(lst):
    last_index = len(lst)-1
    i = 0
    while i <= last_index:
        if i == last_index:
            i = 0
            last_index -= 1                                         #TC - O(n)
        if lst[i] > lst[i+1]:
            lst[i] , lst[i+1] = lst[i+1] , lst[i]
        i += 1
    return lst
print(bubble_sort(lst))
print()


lst = [9,1,2,3,6,88,7,5]
print(lst)

def selection_sort(lst):
    for i in range(len(lst)):
        mini = min(lst[i:])
        ind = lst.index(mini)
        lst[i],lst[ind] = lst[ind],lst[i]
    return lst
print(selection_sort(lst))
print()


def insertion_sort(lst):
    i = 0
    while i < len(lst):
        j = i
        while j > 0:
            if lst[j-1] > lst[j]:
                lst[j-1] , lst[j] = lst[j] , lst[j-1]
            j -= 1
        i += 1
    return lst

lst = [9,1,2,3,6,88,7,5]
print(lst)
print(insertion_sort(lst))
print()



def bucket_sort(lst):
    no_bucket = round(m.sqrt(len(lst)))
    arr = []
    maxi = max(lst)
    for _ in range(no_bucket):
        arr.append([])
    for ele in lst:
        bucket_number = m.ceil((ele*no_bucket)/maxi)
        arr[bucket_number-1].append(ele)
        arr[bucket_number-1].sort()
    while len(arr) != 1:
        arr[0].extend(arr[1])
        arr.pop(1)
    return arr[0]

lst = [9,1,2,3,6,88,7,5,40]
print(lst)
print(bucket_sort(lst))
print()

def merge(customList, lowest, middle, top):
    n1 = middle - lowest + 1
    n2 = top - middle

    Left = [0] * (n1)
    Right = [0] * (n2)

    for i in range(0, n1):
        Left[i] = customList[lowest+i]
    
    for j in range(0, n2):
        Right[j] = customList[middle+1+j]

    i=j=0
    k=lowest

    while i<n1 and j<n2:                    #
        if Left[i] < Right[j]:              #
            customList[k] = Left[i]         #
            i+=1                            #       Sorting Lines of Code
        else:                               #
            customList[k] = Right[j]        #
            j+=1                            #
        k+=1                                #
    
    # Adding Remaining elements of Left or Right (depending on value of i and j)
    while i < n1:
        customList[k] = Left[i]
        i += 1
        k += 1
    
    while j < n2:
        customList[k] = Right[j]
        j += 1
        k += 1

lst = [9,1,2,3,6,88,7,5,40]
print(lst)

def merge_sort(list , l , r):
    if l < r:
        m = (l+(r-1))//2
        merge_sort(list , l , m)
        merge_sort(list , m+1 , r)
        merge(list , l , m , r)
    return list

print(merge_sort(lst , 0 , len(lst)-1))
print()

# Quick Sort
lst = [9,1,2,3,6,88,7,5,40]
print(lst)

def swap(lst , index1 , index2):
    lst[index1] , lst[index2] = lst[index2] , lst[index1]

def pivot(lst , pivot_index , end_index):
    swap_index = pivot_index
    for i in range(pivot_index+1 , end_index+1):
        if lst[i] < lst[pivot_index]:
            swap_index += 1
            swap(lst , swap_index , i)
    swap(lst , pivot_index , swap_index)
    return swap_index

def quicksort(lst , left , right):
    if left<right:
        pivot_index = pivot(lst , left , right)
        quicksort(lst , left , pivot_index-1)
        quicksort(lst , pivot_index+1 , right)
    return lst

print(quicksort(lst,0,len(lst)-1))
print()


lst = [9,1,2,3,6,88,7,5,40]
print(lst)

#Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo 
def heapify(customList, n, i):
    smallest = i
    l = 2*i + 1
    r = 2*i + 2
    if l < n and customList[l] < customList[smallest]:
        smallest = l
    
    if r < n and customList[r] < customList[smallest]:
        smallest = r
    
    if smallest != i:
        customList[i], customList[smallest] = customList[smallest], customList[i]
        heapify(customList, n, smallest)


def heapSort(customList):
    n = len(customList)
    for i in range(int(n/2)-1, -1, -1):
        heapify(customList, n, i)
    
    for i in range(n-1,0,-1):
        customList[i], customList[0] = customList[0], customList[i]
        heapify(customList, i, 0)
    customList.reverse()
heapSort(lst)
print(lst)
#Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo Redo

def findKthLargest(nums, k):
    heapSort(nums)
    return nums[k-1]

nums = [3,2,1,5,6,4]        
print(findKthLargest(nums,2))