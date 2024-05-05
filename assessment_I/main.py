def count_islands(graph):
    def dfs(cur):
        visited.add(cur)
        size = 1
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x, y = cur[0] + dx, cur[1] + dy
            if not (0 <= x < n and 0 <= y < m
                and graph[x][y] == '0'
                and (x,y) not in visited):
                continue
            size += dfs((x, y))
        return size

    visited=set()
    lands = sorted([dfs((i, j)) for i in range(n) for j in range(m)
            if graph[i][j] == '0' and (i, j) not in visited])
    return len(lands), ",".join(map(str, lands))

def best_safe_cost(graph, start, end):
    cost_so_far = {start: 0}
    queue = [(start, 0)]
    while queue:
        cur, cost = queue.pop(0)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x, y = cur[0] + dx, cur[1] + dy
            if not (0 <= x < n and 0 <= y < m):
                continue
            if graph[x][y] in ['0', 's']:
                continue
            new_cost = cost + (0 if graph[x][y] == 'f' else int(graph[x][y]))
            if (x, y) not in cost_so_far or new_cost < cost_so_far[(x, y)]:
                cost_so_far[(x, y)] = new_cost
                queue.append(((x, y), new_cost))

    return cost_so_far[end] if end in cost_so_far else -1

if __name__ == "__main__":
    with open("test_case/ocean.in1", "r") as f:
        n, m = map(int, f.readline().split())
        graph = [list(f.readline().strip()) for _ in range(n)]

    n_islands, islands = count_islands(graph)
    with open("test_case/ocean.out1", "w") as f:
        f.write(str(n_islands) + "\n")
        if n_islands > 0: f.write(islands + ".")

    best_safe_cost = best_safe_cost(graph, (0, 2), (4, 0))
    with open("test_case/ocean.out2", "w") as f:
        f.write(str(best_safe_cost))
