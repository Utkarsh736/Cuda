from heapq import heappush, heappushpop

fn solve(input: ptr[float32], N: Int, k: Int, output: ptr[float32]):
    var heap: List[float32] = []

    for i in range(N):
        let val = input[i]
        if len(heap) < k:
            heappush(heap, val)
        else:
            if val > heap[0]:
                heappushpop(heap, val)

    # Sort in descending order
    heap.sort(reverse=True)

    for i in range(k):
        output[i] = heap[i]
