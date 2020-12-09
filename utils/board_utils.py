import curses
import numpy as np

def num_hidden_boxes(state):
    c = state[0] + state[2]
    n, m = c.shape
    cnt = 0
    for j in range(m):
        hidden = False
        for i in range(n):
            if c[i, j] != 0:
                hidden = True
            if hidden and c[i, j] == 0:
                cnt += 1
    return cnt

def num_hidding_boxes(state):
    c = state[0] + state[2]
    n, m = c.shape
    cnt = 0
    for j in range(m):
        hidden = False
        for i in range(n-1, -1, -1):
            if c[i, j] == 0:
                hidden = True
            if hidden and c[i, j] != 0:
                cnt += 1
    return cnt

def num_closed_boxes(state):
    c = state[0] + state[2]
    n, m = c.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()
    queue = [(0, j) for j in range(m) if c[0, j] == 0]
    while len(queue) > 0:
        u = queue.pop()
        for d in directions:
            v = (u[0] + d[0], u[1] + d[1])
            if v[0] < 0 or v[0] >= n or v[1] < 0 or v[1] >= m:
                continue
            if c[v] == 0 and v not in visited:
                visited.add(v)
                queue.append(v)
    closed = n * m - c.sum() - len(visited)
    return closed

def num_closed_regions(state):
    c = state[0] + state[2]
    n, m = c.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()
    region_cnt = 0
    for i in range(n):
        for j in range(m):
            if (i, j) not in visited:
                region_cnt += 1
                queue = [(i, j)]
                visited.add((i, j))
                while len(queue) > 0:
                    u = queue.pop()
                    for d in directions:
                        v = (u[0] + d[0], u[1] + d[1])
                        if v[0] < 0 or v[0] >= n or v[1] < 0 or v[1] >= m:
                            continue
                        if c[v] == 0 and v not in visited:
                            visited.add(v)
                            queue.append(v)
    return region_cnt


def num_shared_edges(state):
    c, h = state[0], state[2]
    n, m = c.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    cnt = 0
    for i in range(n):
        for j in range(m):
            if c[i, j] != 0:
                for d in directions:
                    ii, jj = i + d[0], j + d[1]
                    if ii < 0 or ii >= n or jj < 0 or jj >= m:
                        cnt += 1  # shared with boarder
                        continue
                    if c[ii, jj] != 0:
                        cnt += 1
    return cnt

def board_height(state):
    c = state[0] + state[2]
    n, m = c.shape
    for i in range(n):
        for j in range(m):
            if c[i, j] != 0:
                return n - i
    return 0

def board_box_height(state, scores=None):
    c = state[0] + state[2]
    n, m = c.shape
    if scores is None:
        scores = [d ** 1.5 / 10.0 for d in range(n)]
    score = 0
    for i in range(n):
        for j in range(m):
            if c[i, j] != 0:
                score += scores[i]
    return score

def board_line_score(state, scores=None):
    c = state[0] + state[2]
    n, m = c.shape
    if scores is None:
        scores = [0] * 11
        scores[-1] = 200
        for i in range(10):
            scores[i] = i ** 1.5
    score = 0
    for i in range(n):
        score += scores[c[i].sum()]
    return score
    
def hidden_boxes_score(c, k=-0.3, p=1):
    return k * num_hidden_boxes(c) ** p

def hidding_boxes_score(c, k=-0.3, p=1):
    return k * num_hidding_boxes(c) ** p

def closed_boxes_score(c, k=-0.5, p=1):
    return k * num_closed_boxes(c) ** p

def closed_regions_score(c, k=-3.0, p=1):
    return k * num_closed_regions(c) ** p

def shared_edges_score(c, k=0.1, p=1):
    return k * num_shared_edges(c) ** p

def board_height_score(c, k=-1.0, p=1.0):
    return k * board_height(c) ** p

def boxes_in_a_line_score(c, k=0.01, p=3):
    return board_line_score(c, scores=None)
    # return board_line_score(c, scores=[k * i ** p for i in range(0, 11)])

def board_box_height_score(c, k=0.01, p=1.5):
    return board_box_height(c, scores=[k * d ** p for d in range(c.shape[1])])

# transition
def penalize_hidden_boxes(p, c):
    if c is None: 
        return 0.0
    return hidden_boxes_score(c) - hidden_boxes_score(p)

def penalize_hidding_boxes(p, c):
    if c is None: 
        return 0.0
    return hidding_boxes_score(c) - hidding_boxes_score(p)

def penalize_closed_boxes(p, c):
    if c is None: 
        return 0.0
    return closed_boxes_score(c) - closed_boxes_score(p)

def penalize_closed_regions(p, c):
    if c is None: 
        return 0.0
    return closed_regions_score(c) - closed_regions_score(p)

def encourage_shared_edges(p, c):
    if c is None: 
        return 0.0
    return shared_edges_score(c) - shared_edges_score(p)

def penalize_higher_boxes(p, c):
    if c is None: 
        return 0.0
    return board_height_score(c) - board_height_score(p)

def encourage_boxex_in_a_line(p, c):
    if c is None: 
        return 0.0
    return boxes_in_a_line_score(c) - boxes_in_a_line_score(p)

def encourage_lower_layers(p, c):
    if c is None: 
        return 0.0
    return board_box_height_score(c) - board_box_height_score(p)

def print_observation(ob, stdcsr=None):
    if ob is None:
        ob = np.zeros(shape=(3, 21, 10), dtype=np.int)
    n, m = ob[0].shape
    if stdcsr:
        stdcsr.clear()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
    for i in range(n+2):
        for j in range(m+2):
            clr = 1
            if (i == 0 or i == n+1) and (j == 0 or j == m+1):
                pch = '+'
                if stdcsr:
                    if i == 0 and j == 0:
                        ch = curses.ACS_ULCORNER
                    elif i == 0 and j == m+1:
                        ch = curses.ACS_URCORNER
                    elif i == n+1 and j == 0:
                        ch = curses.ACS_LLCORNER
                    else:
                        ch = curses.ACS_LRCORNER
            elif i == 0 or i == n+1:
                pch = '-'
                if stdcsr:
                    ch = curses.ACS_HLINE
            elif j == 0 or j == m+1:
                pch = '|'
                if stdcsr:
                    ch = curses.ACS_VLINE
            else:
                if ob[0, i-1, j-1] != 0:
                    pch = '.'
                    if stdcsr:
                        ch = curses.ACS_BLOCK
                elif ob[1, i-1, j-1] != 0:
                    pch = '~'
                    if stdcsr:
                        ch = curses.ACS_BLOCK
                        clr = 2
                elif ob[2, i-1, j-1] != 0:
                    pch = 'x'
                    if stdcsr:
                        ch = curses.ACS_BLOCK
                        clr = 3
                else:
                    pch = ' '
                    ch = ' '
            if stdcsr:
                stdcsr.addch(ch, curses.color_pair(clr))
            else:
                print(pch, end="")
        if stdcsr:
            stdcsr.addstr('\n')
        else:
            print()
    score_functions = {
        'Board Height': board_height_score,
        'Hidden Boxes': hidden_boxes_score,
        'Hidding Boxes': hidding_boxes_score,
        'Closed Boxes': closed_boxes_score,
        'Closed Regiones': closed_regions_score,
        'Shared Edges': shared_edges_score,
        'Boxes in a Line': boxes_in_a_line_score,
        'Board Box Height': board_box_height_score,
    }
    count_functions = {
        'Height': board_height,
        'Hidden Boxes': num_hidden_boxes,
        'Hidding Boxes': num_hidding_boxes,
        'Closed Boxes': num_closed_boxes,
        'Closed Boxes': num_closed_regions,
        'Shared edges': num_shared_edges
    }
    summary_lines = []
    summary_lines.append(f'{"Counter":>20}:')
    summary_lines.extend([f'{name:>20}: {func(ob)}' for name, func in count_functions.items()])
    summary_lines.append(f'{"Score":>20}:')
    summary_lines.extend([f'{name:>20}: {func(ob):.1f}' for name, func in score_functions.items()])
    summary_str = "\n".join(summary_lines)
    if stdcsr:
        stdcsr.addstr(summary_str + '\n')
    else:
        print(summary_str)
