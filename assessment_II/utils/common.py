import pandas as pd
import numpy as np
from collections import defaultdict

def list_way(n):
    """i < n/2: i even x j odd, i >=n/2: i odd x j even"""
    l = (
        [(0, j) for j in range(n+2)]
        + [(i, j) for j in range(n+2)
                for i in range(1, n//2)
                if not ((i % 2 == 0 and j % 2 == 1) or (i == j)) or j == n+1]
        + [(i, j) for j in range(n+2)
                for i in range(n//2, n+1)
                if not ((i % 2 == 1 and j % 2 == 0) or (i == j)) or j == n+1]
        + [(n+1, j) for j in range(n+2)]
        )
    return np.array(l)

def create_mask(n):
    visited = np.zeros((n+2,n+2), dtype=bool)
    l = list_way(n)
    visited[l[:, 0], l[:, 1]] = True
    visited[:, 0] = False
    visited[np.eye(n+2, dtype=bool)] = False
    return visited

def create_matrix_csv(file):
    df = pd.read_csv(file)
    coords = np.array([(x, y) for x, y in zip(df['Var2'], df['Var3'])])
    n = len(coords) - 2
    x = coords[:, 0]
    y = coords[:, 1]
    xdiff = x[:, None] - x[None, :]
    ydiff = y[:, None] - y[None, :]
    m = np.sqrt(xdiff**2+ydiff**2)
    visited = create_mask(n)
    m[~visited] = 0
    return m

def calc_custom(m, tour):
    """Calculate the cost of a given tour"""
    dists = m[tour[:-1], :][:, tour[1:]].diagonal()
    total_distance = np.sum(dists)
    max_dist = np.max(dists)
    delta = max_dist - np.min(dists)
    total_cost = (len(tour) - 2)*delta*max_dist+ total_distance
    return total_cost

def calc_distance(m, tour):
    """Calculate the cost of a given tour"""
    dists = m[tour[:-1], :][:, tour[1:]].diagonal()
    total_distance = np.sum(dists)
    return total_distance

def create_weight_matrix(m, n):
    m_discount = m.copy()
    for i in range(n+2):
        for j in range(n+2):
            if m_discount[i][j] == 0:
                continue
            if i % 2 == 0 and j % 2 == 0: # nếu là chẵn
                if i % 2 < n//2 and j % 2 < n//2: # chẵn nhỏ-chẵn nhỏ
                    m_discount[i][j] -= m_discount[i][j]*0.15
                else:
                    m_discount[i][j] -= m_discount[i][j]*0.1
            if i % 2 == 1 and j % 2 == 1: # nếu là lẻ
                if i % 2 >= n//2 and j % 2 >= n//2: #lẻ lớn - lẻ lớn
                    m_discount[i][j] -= m_discount[i][j]*0.15
                else:
                    m_discount[i][j] -= m_discount[i][j]*0.1
    return m_discount
