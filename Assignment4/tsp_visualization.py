import math
import random
import heapq
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # use a non‑interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Number of cities in the generated graph.  The user requested to
# increase this to around 15.  Higher values dramatically increase the
# computational cost of exhaustive search algorithms like backtracking,
# so be cautious if increasing further.
NUM_CITIES = 8

# Range for random coordinate generation.  The cities are placed
# uniformly within a square of side length COORD_RANGE.  Distances are
# computed as Euclidean distances between these coordinates.
COORD_RANGE = 100.0

# Seed for reproducibility.  If None, a random seed is generated.  The
# seed used is printed when the script runs so that the dataset and
# resulting animation may be exactly reproduced.
RANDOM_SEED = 20009

# How many search events to record from each algorithm.  The full
# algorithms explore thousands of nodes; capturing every event would
# result in an unmanageably long animation.  Instead, we record a
# representative prefix of the event stream while still executing the
# algorithms completely to obtain the correct optimal tour and counts.
MAX_EVENTS_PER_ALGO = float('inf')

# Visual styling constants.  These values govern the colours, marker
# sizes, line widths and fonts used throughout the animation.  Feel
# free to experiment with these to obtain a pleasant visual appearance.
BACKGROUND_COLOUR = "#1e1e2e"  # dark background for good contrast
NODE_COLOUR = "#8aadf4"        # base colour for city nodes
START_NODE_COLOUR = "#f4b942"  # highlight colour for the starting city
VISITED_NODE_COLOUR = "#a6da95"  # colour for visited nodes during search
CURRENT_NODE_COLOUR = "#ed8796"  # colour for the node currently being expanded
EDGE_COLOUR = "#44475a"         # muted edge colour
PATH_COLOUR_BT = "#8aadf4"      # colour for backtracking edges
PATH_COLOUR_BNB = "#f5a97f"     # colour for branch & bound edges
BEST_PATH_COLOUR = "#f5e0dc"    # colour for the final optimal tour
PRUNE_COLOUR = "#ed8796"        # colour used to mark pruned branches

# Frame timing.  These numbers determine how many frames are inserted
# for various animation actions.  See the user instructions for
# guidance on their relative sizes.  The sum of the smooth, pause and
# highlight frames per event will govern the total length of the GIF.
SMOOTH_FRAMES = 13   # frames used to smoothly draw an edge or fade text
PAUSE_FRAMES = 6     # frames to hold after each event so it can be seen
HIGHLIGHT_FRAMES = 8  # additional frames when highlighting a newly found best path
TEXT_FADE_FRAMES = 6  # frames used to fade in/out annotation text
TEXT_HOLD_FRAMES = 8  # frames to hold annotation text fully visible
LEGEND_FADE_FRAMES = 8  # frames for legend fade in/out

# Animation settings
FPS = 24  # frames per second for the final GIF
FIGSIZE = (6, 6)  # 6 inches square figure
RESOLUTION = 120  # 6in × 120dpi → 720px, satisfying the ≥720p requirement


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_random_graph(n: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random set of n city positions and the corresponding
    symmetric distance matrix.

    The positions are sampled uniformly within [0, COORD_RANGE]².  The
    distance matrix contains Euclidean distances between the points,
    rounded to two decimals to avoid tiny floating point artefacts.

    Parameters
    ----------
    n : int
        Number of cities to generate.  Must be at least 2.
    seed : int, optional
        Random seed for reproducibility.  If None, a fresh seed is
        generated.

    Returns
    -------
    positions : np.ndarray of shape (n, 2)
        2D coordinates for each city.
    distances : np.ndarray of shape (n, n)
        Symmetric matrix of pairwise distances.
    """
    if n < 2:
        raise ValueError("Number of cities must be at least 2")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        # Generate a random seed and set it
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
    # Sample positions
    positions = np.random.rand(n, 2) * COORD_RANGE
    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            distances[i, j] = distances[j, i] = round(d, 2)
    return positions, distances


# ---------------------------------------------------------------------------
# Backtracking TSP solver with event recording
# ---------------------------------------------------------------------------

def solve_tsp_backtracking(distances: np.ndarray, max_events: int) -> Tuple[List[int], float, int, List[Dict[str, Any]]]:
    """Solve the travelling salesman problem using a simple backtracking
    approach.  A subset of search events is recorded for animation.

    This algorithm explores Hamiltonian cycles by expanding nodes one
    after another from the start city (index 0).  At each step it
    prunes branches whose cost exceeds the best solution found so far.
    It records up to ``max_events`` events describing the first part
    of the search for later visualisation.  The algorithm still
    computes the full optimal solution and exploration count even if
    the event list is truncated.

    Parameters
    ----------
    distances : np.ndarray
        Symmetric matrix of pairwise distances between cities.
    max_events : int
        Maximum number of events to record for animation.  If this
        limit is reached, further events are not recorded but the
        search continues silently.

    Returns
    -------
    best_path : List[int]
        The optimal Hamiltonian cycle (starting at 0, excluding the
        final return to 0).
    best_cost : float
        Total cost of the optimal cycle.
    expansions : int
        Number of partial solutions explored (including pruned ones).
    events : List[Dict[str, Any]]
        Recorded search events for visualisation.
    """
    n = distances.shape[0]
    best_cost = float('inf')
    best_path = []
    visited = [False] * n
    visited[0] = True
    events: List[Dict[str, Any]] = []
    expansions = 0

    def record_event(event: Dict[str, Any]):
        """Append an event to the list if the limit has not been reached."""
        if len(events) < max_events:
            events.append(event)

    def dfs(path: List[int], cost_so_far: float):
        nonlocal best_cost, best_path, expansions
        # If all cities have been visited, complete the tour
        if len(path) == n:
            expansions += 1
            total_cost = cost_so_far + distances[path[-1]][path[0]]
            # Record the completion
            record_event({
                'algorithm': 'BT',
                'type': 'complete',
                'path': path.copy(),
                'cost_so_far': cost_so_far,
                'total_cost': total_cost,
                'best_cost': best_cost,
                'expansions': expansions
            })
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path.copy()
                record_event({
                    'algorithm': 'BT',
                    'type': 'update_best',
                    'path': path.copy(),
                    'total_cost': total_cost,
                    'best_cost': best_cost,
                    'expansions': expansions
                })
            return
        # Try expanding to each unvisited city
        for next_city in range(1, n):
            if not visited[next_city]:
                new_cost = cost_so_far + distances[path[-1]][next_city]
                expansions += 1
                record_event({
                    'algorithm': 'BT',
                    'type': 'explore',
                    'path': path + [next_city],
                    'cost_so_far': cost_so_far,
                    'new_cost': new_cost,
                    'best_cost': best_cost,
                    'expansions': expansions
                })
                if new_cost < best_cost:
                    visited[next_city] = True
                    dfs(path + [next_city], new_cost)
                    visited[next_city] = False
                else:
                    # This branch is pruned because its cost already exceeds the best known
                    record_event({
                        'algorithm': 'BT',
                        'type': 'prune',
                        'path': path + [next_city],
                        'cost_so_far': cost_so_far,
                        'new_cost': new_cost,
                        'best_cost': best_cost,
                        'expansions': expansions
                    })
                # Record backtrack after exploring each child
                record_event({
                    'algorithm': 'BT',
                    'type': 'backtrack',
                    'path': path.copy(),
                    'cost_so_far': cost_so_far,
                    'best_cost': best_cost,
                    'expansions': expansions
                })

    # Record initial state
    record_event({
        'algorithm': 'BT',
        'type': 'start',
        'path': [0],
        'cost_so_far': 0.0,
        'best_cost': best_cost,
        'expansions': expansions
    })
    dfs([0], 0.0)
    return best_path, best_cost, expansions, events


# ---------------------------------------------------------------------------
# Branch and Bound TSP solver with event recording
# ---------------------------------------------------------------------------

def solve_tsp_branch_and_bound(distances: np.ndarray, max_events: int) -> Tuple[List[int], float, int, List[Dict[str, Any]]]:
    """Solve the travelling salesman problem using branch and bound.

    A simple bounding function is used which sums the minimum outgoing edge
    cost from each unvisited city.  While not particularly tight, this
    heuristic suffices to illustrate how branch and bound prioritises
    promising branches and prunes unpromising ones.  As with the
    backtracking solver, only a limited prefix of the search events is
    recorded for animation purposes.

    Parameters
    ----------
    distances : np.ndarray
        Symmetric matrix of pairwise distances between cities.
    max_events : int
        Maximum number of events to record for animation.  Additional
        search steps are executed without being recorded.

    Returns
    -------
    best_path : List[int]
        Optimal Hamiltonian cycle (starting at 0).
    best_cost : float
        Total cost of the optimal cycle.
    expansions : int
        Number of partial solutions expanded (including pruned ones).
    events : List[Dict[str, Any]]
        Recorded events for visualisation.
    """
    n = distances.shape[0]
    # Precompute the minimal outgoing edge cost for each city (excluding self)
    min_outgoing = [min(distances[i][j] for j in range(n) if j != i) for i in range(n)]

    def bound(path: List[int], cost_so_far: float, visited_mask: List[bool]) -> float:
        """Compute a simple lower bound on the completion cost of a partial path.
        The bound equals the cost accumulated so far plus the sum of minimal
        outgoing edges for all unvisited cities, plus the minimal outgoing
        edge from the last city in the path.  This is a coarse bound
        designed for illustrative purposes.
        """
        b = cost_so_far
        last = path[-1]
        # cost to leave last city
        b += min_outgoing[last]
        # cost to visit all unvisited cities
        for city in range(n):
            if not visited_mask[city]:
                b += min_outgoing[city]
        return b

    events: List[Dict[str, Any]] = []
    expansions = 0
    best_cost = float('inf')
    best_path: List[int] = []

    def record_event(event: Dict[str, Any]):
        if len(events) < max_events:
            events.append(event)

    # Priority queue of (bound, cost_so_far, path, visited_mask)
    start_path = [0]
    visited_mask = [False] * n
    visited_mask[0] = True
    initial_bound = bound(start_path, 0.0, visited_mask)
    queue: List[Tuple[float, float, List[int], List[bool]]] = []
    heapq.heappush(queue, (initial_bound, 0.0, start_path, visited_mask.copy()))

    record_event({
        'algorithm': 'BnB',
        'type': 'start',
        'path': start_path.copy(),
        'cost_so_far': 0.0,
        'bound': initial_bound,
        'best_cost': best_cost,
        'expansions': expansions
    })

    while queue:
        current_bound, cost_so_far, path, visited_mask = heapq.heappop(queue)
        expansions += 1
        # If the bound already exceeds the best known cost, prune this branch
        if current_bound >= best_cost:
            record_event({
                'algorithm': 'BnB',
                'type': 'prune',
                'path': path.copy(),
                'cost_so_far': cost_so_far,
                'bound': current_bound,
                'best_cost': best_cost,
                'expansions': expansions
            })
            continue
        # If all cities have been visited, complete the tour
        if len(path) == n:
            total_cost = cost_so_far + distances[path[-1]][path[0]]
            record_event({
                'algorithm': 'BnB',
                'type': 'complete',
                'path': path.copy(),
                'cost_so_far': cost_so_far,
                'bound': current_bound,
                'total_cost': total_cost,
                'best_cost': best_cost,
                'expansions': expansions
            })
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path.copy()
                record_event({
                    'algorithm': 'BnB',
                    'type': 'update_best',
                    'path': path.copy(),
                    'total_cost': total_cost,
                    'best_cost': best_cost,
                    'expansions': expansions
                })
            continue
        # Expand each unvisited city
        last_city = path[-1]
        for next_city in range(1, n):
            if not visited_mask[next_city]:
                new_cost = cost_so_far + distances[last_city][next_city]
                # Compute bound for the new partial path
                visited_mask_next = visited_mask.copy()
                visited_mask_next[next_city] = True
                b = bound(path + [next_city], new_cost, visited_mask_next)
                # Record exploration event
                record_event({
                    'algorithm': 'BnB',
                    'type': 'explore',
                    'path': path + [next_city],
                    'cost_so_far': cost_so_far,
                    'new_cost': new_cost,
                    'bound': b,
                    'best_cost': best_cost,
                    'expansions': expansions
                })
                # Only push to queue if the bound is promising
                heapq.heappush(queue, (b, new_cost, path + [next_city], visited_mask_next))

    return best_path, best_cost, expansions, events


# ---------------------------------------------------------------------------
# Animation helper class
# ---------------------------------------------------------------------------

class TSPAnimation:
    """A helper to construct a smooth animated GIF depicting the search
    process of two TSP algorithms.

    Given the positions of the cities, the event streams from the
    backtracking and branch‑and‑bound solvers, and the corresponding
    optimal solutions, this class generates an animation composed of
    discrete frames.  Each event contributes a number of frames to
    represent the action, including smooth line drawing, text fade in
    and hold frames.  The resulting frames are converted to PIL images
    and saved as a single GIF file.
    """

    def __init__(self,
                 positions: np.ndarray,
                 distances: np.ndarray,
                 bt_events: List[Dict[str, Any]],
                 bnb_events: List[Dict[str, Any]],
                 bt_best_path: List[int],
                 bt_best_cost: float,
                 bnb_best_path: List[int],
                 bnb_best_cost: float,
                 bt_expansions: int,
                 bnb_expansions: int):
        self.positions = positions
        self.distances = distances
        self.bt_events = bt_events
        self.bnb_events = bnb_events
        self.bt_best_path = bt_best_path
        self.bt_best_cost = bt_best_cost
        self.bnb_best_path = bnb_best_path
        self.bnb_best_cost = bnb_best_cost
        self.bt_expansions = bt_expansions
        self.bnb_expansions = bnb_expansions
        # Precompute node indices for convenience
        self.n = positions.shape[0]
        # Create a font for annotation text using a default sans‑serif font
        # Use a TrueType font for richer Unicode support (e.g. the infinity symbol)
        # and better aesthetics.  We look up the DejaVu Sans font shipped with
        # matplotlib.  If this lookup fails, we fall back to PIL's default.
        try:
            # Attempt to load a font capable of displaying Chinese characters.  We
            # try SimHei first (common on Windows), then Microsoft YaHei, then
            # fall back to DejaVu Sans.  If none of these are found we use
            # PIL's default bitmap font.
            from matplotlib import font_manager
            font_path = None
            for fname in [
                "SimHei",                # 常见中文黑体（Windows）
                "Microsoft YaHei",       # 微软雅黑（Windows）
                "Noto Sans CJK SC",      # Noto Sans 简体中文
                "NotoSansCJK-Regular",   # Noto Sans CJK fallback
                "DejaVu Sans"            # 默认字体
            ]:
                try:
                    font_path_candidate = font_manager.findfont(fname)
                    if font_path_candidate and os.path.isfile(font_path_candidate):
                        font_path = font_path_candidate
                        break
                except Exception:
                    continue
            if font_path and os.path.isfile(font_path):
                # If the selected font is DejaVu Sans and a CJK font is available, prefer the CJK font for
                # proper Chinese character rendering.  DejaVu Sans lacks CJK glyphs on many systems.
                if 'DejaVuSans' in os.path.basename(font_path).replace('-', ''):
                    noto_candidate = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
                    if os.path.exists(noto_candidate):
                        font_path = noto_candidate
                self.font_large = ImageFont.truetype(font_path, int(RESOLUTION * 0.2))
                self.font_small = ImageFont.truetype(font_path, int(RESOLUTION * 0.1))
            else:
                # As a final fallback, attempt to use NotoSansCJK directly via absolute path
                noto_candidate = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
                if os.path.exists(noto_candidate):
                    self.font_large = ImageFont.truetype(noto_candidate, int(RESOLUTION * 0.2))
                    self.font_small = ImageFont.truetype(noto_candidate, int(RESOLUTION * 0.1))
                else:
                    # Default to PIL's bitmap font if no suitable TTF is found
                    self.font_large = ImageFont.load_default()
                    self.font_small = ImageFont.load_default()
        except Exception:
            self.font_large = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
        # Determine figure size in pixels
        width, height = int(FIGSIZE[0] * RESOLUTION), int(FIGSIZE[1] * RESOLUTION)
        self.figsize_px = (width, height)

    def _draw_graph(self, ax: plt.Axes, visited_nodes: List[bool] = None,
                    current_path: List[int] = None,
                    algorithm: str = '') -> None:
        """Draw the base graph on the provided axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw on.
        visited_nodes : list of bool, optional
            A boolean list indicating which nodes have been visited.  If
            provided, visited nodes are coloured differently.
        current_path : list of int, optional
            The current path being explored.  Its edges are drawn in a
            distinct colour according to the algorithm.
        algorithm : str, optional
            Either 'BT' or 'BnB', controlling the colour of the path.
        """
        # Draw all edges lightly
        for i in range(self.n):
            for j in range(i + 1, self.n):
                xi, yi = self.positions[i]
                xj, yj = self.positions[j]
                ax.plot([xi, xj], [yi, yj], color=EDGE_COLOUR, linewidth=1, zorder=1)
        # Draw nodes
        for idx, (x, y) in enumerate(self.positions):
            if visited_nodes is not None and visited_nodes[idx]:
                node_colour = VISITED_NODE_COLOUR
            else:
                node_colour = NODE_COLOUR
            if idx == 0:
                node_colour = START_NODE_COLOUR
            ax.add_patch(Circle((x, y), 2.5, color=node_colour, zorder=3))
            # Draw label slightly offset
            ax.text(x + 1.5, y + 1.5, str(idx), color="#ffffff", fontsize=13, zorder=4)
        # Draw current path
        if current_path and len(current_path) > 1:
            colour = PATH_COLOUR_BT if algorithm == 'BT' else PATH_COLOUR_BNB
            for i in range(len(current_path) - 1):
                a = current_path[i]
                b = current_path[i + 1]
                xi, yi = self.positions[a]
                xj, yj = self.positions[b]
                ax.plot([xi, xj], [yi, yj], color=colour, linewidth=2.5, zorder=2)

    def _init_canvas(self) -> Tuple[plt.Figure, plt.Axes]:
        """Initialise a matplotlib figure and axes with the chosen theme."""
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=RESOLUTION)
        ax.set_facecolor(BACKGROUND_COLOUR)
        fig.patch.set_facecolor(BACKGROUND_COLOUR)
        # Remove axis ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig, ax

    def _convert_fig_to_image(self, fig: plt.Figure) -> Image.Image:
        """Convert a matplotlib figure to a PIL Image and close the figure.

        The figure is drawn to the Agg backend, converted to an RGBA
        image and then closed to free resources.  Closing the figure
        promptly prevents a buildup of open figures which would
        otherwise emit warnings and consume memory.
        """
        try:
            fig.canvas.draw()
            w, h = self.figsize_px
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (h, w, 4)
            rgba = np.roll(buf, -1, axis=2)
            image = Image.fromarray(rgba, mode='RGBA')
        finally:
            plt.close(fig)
        return image

    def _draw_text_overlay(self, image: Image.Image, lines: List[str], opacity: float) -> Image.Image:
        """Draw semi‑transparent text overlay onto an image.

        Parameters
        ----------
        image : PIL.Image
            The base image.
        lines : list of str
            Lines of text to draw.  They will be stacked vertically.
        opacity : float
            Opacity of the text, between 0 (invisible) and 1 (fully opaque).

        Returns
        -------
        PIL.Image
            A new image with the text drawn.
        """
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        # Determine line height using the large font.  Pillow >=10 removed
        # getsize() from FreeTypeFont; getbbox() returns (left, top, right, bottom).
        try:
            bbox = self.font_large.getbbox("Mg")
            line_height = bbox[3] - bbox[1]
        except Exception:
            # Fallback: approximate height
            line_height = int(RESOLUTION * 0.06)
        x = 10
        y = 10
        # Draw each line with a slight shadow for readability
        for line in lines:
            # Shadow
            draw.text((x + 1, y + 1), line, font=self.font_large, fill=(0, 0, 0, int(opacity * 255 * 0.6)))
            draw.text((x, y), line, font=self.font_large, fill=(255, 255, 255, int(opacity * 255)))
            y += line_height + 4
        return Image.alpha_composite(image, overlay)

    def build_frames(self) -> List[Image.Image]:
        """Construct the entire list of frames for the animation.

        The animation sequence is organised as follows:

        1. Fade in the legend and initial title.
        2. Show the graph structure with no search progress.
        3. Play through a limited number of search events for the backtracking
           algorithm, smoothly drawing edges and updating annotations.
        4. Summarise the backtracking result and hold for a few seconds.
        5. Repeat steps 3–4 for the branch & bound algorithm.
        6. Present a final comparison of the two methods.

        Returns
        -------
        List[PIL.Image]
            A list of frames ready to be saved as a GIF.
        """
        frames: List[Image.Image] = []

        # ------------------------------------------------------------------
        # Phase 1: Initial legend fade in
        # ------------------------------------------------------------------
        fig, ax = self._init_canvas()
        self._draw_graph(ax)
        # Create a dummy image to overlay legend
        base_img = self._convert_fig_to_image(fig)
        legend_lines = [
            "旅行售货员问题", 
            f"城市数量：{self.n}（随机种子：{RANDOM_SEED}）",
            "算法：回溯法 vs 分支限界法"
        ]
        for i in range(LEGEND_FADE_FRAMES):
            opacity = (i + 1) / LEGEND_FADE_FRAMES
            img = base_img.copy()
            img = self._draw_text_overlay(img, legend_lines, opacity)
            frames.append(img)
        # Hold the legend fully visible
        for _ in range(TEXT_HOLD_FRAMES):
            frames.append(self._draw_text_overlay(base_img.copy(), legend_lines, 1.0))

        # ------------------------------------------------------------------
        # Phase 2: Show the base graph clearly for a moment
        # ------------------------------------------------------------------
        for _ in range(PAUSE_FRAMES):
            frames.append(base_img.copy())

        # Helper function to animate one algorithm's events
        def animate_algorithm(events: List[Dict[str, Any]],
                              best_path: List[int],
                              best_cost: float,
                              expansions: int,
                              label: str):
            # Map internal algorithm labels to human‑readable Chinese name
            name_map = {'BT': '回溯法', 'BnB': '分支限界法'}
            nonlocal frames
            countframestart=len(frames)
            visited = [False] * self.n
            current_path: List[int] = []
            # Title for this phase
            phase_title = [f"{name_map.get(label, label)} 搜索"]
            # Draw initial frame with title
            fig, ax = self._init_canvas()
            self._draw_graph(ax, visited_nodes=visited, current_path=current_path, algorithm=label)
            base = self._convert_fig_to_image(fig)
            # Fade in title
            for i in range(TEXT_FADE_FRAMES):
                opacity = (i + 1) / TEXT_FADE_FRAMES
                img = base.copy()
                img = self._draw_text_overlay(img, phase_title, opacity)
                frames.append(img)
            # Hold title
            for _ in range(TEXT_HOLD_FRAMES):
                frames.append(self._draw_text_overlay(base.copy(), phase_title, 1.0))
            # Process each event
            for ev in events:
                etype = ev['type']
                if etype == 'start':
                    current_path = ev['path']
                    visited = [False] * self.n
                    visited[current_path[0]] = True
                    # Draw start frame
                    fig, ax = self._init_canvas()
                    self._draw_graph(ax, visited_nodes=visited, current_path=current_path, algorithm=label)
                    frame = self._convert_fig_to_image(fig)
                    # Fade in algorithm name and state
                    lines = [f"{name_map.get(label, label)}：开始探索", f"路径：{current_path}", f"当前最优：∞"]
                    for i in range(TEXT_FADE_FRAMES):
                        op = (i + 1) / TEXT_FADE_FRAMES
                        frames.append(self._draw_text_overlay(frame.copy(), lines, op))
                    # Hold
                    for _ in range(PAUSE_FRAMES):
                        frames.append(self._draw_text_overlay(frame.copy(), lines, 1.0))
                elif etype == 'explore':
                    new_path = ev['path']
                    # Determine the new city visited
                    if len(new_path) > len(current_path):
                        new_city = new_path[-1]
                        visited[new_city] = True
                    current_path = new_path
                    # Draw path incrementally over SMOOTH_FRAMES frames
                    for i in range(SMOOTH_FRAMES):
                        fig, ax = self._init_canvas()
                        # Determine partial edge length to draw
                        # Draw base graph and nodes
                        self._draw_graph(ax, visited_nodes=visited, current_path=current_path[:-1], algorithm=label)
                        # Draw the growing last edge
                        if len(current_path) > 1:
                            a = current_path[-2]
                            b = current_path[-1]
                            xi, yi = self.positions[a]
                            xj, yj = self.positions[b]
                            # Interpolate
                            t = (i + 1) / SMOOTH_FRAMES
                            xt = xi + (xj - xi) * t
                            yt = yi + (yj - yi) * t
                            ax.plot([xi, xt], [yi, yt], color=(PATH_COLOUR_BT if label == 'BT' else PATH_COLOUR_BNB), linewidth=2.5, zorder=2)
                        frame = self._convert_fig_to_image(fig)
                        # Compose annotation
                        lines = [
                            f"{name_map.get(label, label)}：探索", 
                            f"路径：{current_path}",
                            f"代价：{ev.get('new_cost', ev.get('cost_so_far', 0)):.2f}",
                            f"当前最优：{ev['best_cost']:.2f}" if math.isfinite(ev['best_cost']) else "当前最优：∞"
                        ]
                        frames.append(self._draw_text_overlay(frame, lines, 1.0))
                    # Pause after completing the edge
                    for _ in range(PAUSE_FRAMES):
                        fig, ax = self._init_canvas()
                        self._draw_graph(ax, visited_nodes=visited, current_path=current_path, algorithm=label)
                        frame = self._convert_fig_to_image(fig)
                        lines = [
                            f"{name_map.get(label, label)}：探索", 
                            f"路径：{current_path}",
                            f"代价：{ev.get('new_cost', ev.get('cost_so_far', 0)):.2f}",
                            f"当前最优：{ev['best_cost']:.2f}" if math.isfinite(ev['best_cost']) else "当前最优：∞"
                        ]
                        frames.append(self._draw_text_overlay(frame, lines, 1.0))
                elif etype == 'prune':
                    # Mark pruned path with a flash of red colour
                    prune_path = ev['path']
                    # Draw the graph with the pruned edge highlighted
                    for i in range(PAUSE_FRAMES):
                        fig, ax = self._init_canvas()
                        self._draw_graph(ax, visited_nodes=visited, current_path=current_path, algorithm=label)
                        # Draw the pruned edge in prune colour
                        a = prune_path[-2]
                        b = prune_path[-1]
                        xi, yi = self.positions[a]
                        xj, yj = self.positions[b]
                        ax.plot([xi, xj], [yi, yj], color=PRUNE_COLOUR, linewidth=2.0, linestyle='--', zorder=2)
                        frame = self._convert_fig_to_image(fig)
                        lines = [
                            f"{name_map.get(label, label)}：剪枝", 
                            f"路径：{prune_path}",
                            f"代价：{ev.get('new_cost', ev.get('cost_so_far', 0)):.2f}",
                            f"当前最优：{ev['best_cost']:.2f}" if math.isfinite(ev['best_cost']) else "当前最优：∞"
                        ]
                        frames.append(self._draw_text_overlay(frame, lines, 1.0))
                elif etype == 'backtrack':
                    # Remove the last node from current path when backtracking
                    if len(current_path) > 1:
                        last_node = current_path[-1]
                        visited[last_node] = False
                        current_path = current_path[:-1]
                    # Draw backtrack indication
                    fig, ax = self._init_canvas()
                    self._draw_graph(ax, visited_nodes=visited, current_path=current_path, algorithm=label)
                    frame = self._convert_fig_to_image(fig)
                    lines = [
                        f"{name_map.get(label, label)}：回溯", 
                        f"路径：{current_path}",
                        f"当前最优：{ev['best_cost']:.2f}" if math.isfinite(ev['best_cost']) else "当前最优：∞"
                    ]
                    for _ in range(PAUSE_FRAMES):
                        frames.append(self._draw_text_overlay(frame, lines, 1.0))
                elif etype == 'complete':
                    # Completed a tour; highlight it temporarily
                    comp_path = ev['path']
                    total_cost = ev['total_cost']
                    # Draw the complete cycle
                    fig, ax = self._init_canvas()
                    # Mark all nodes visited
                    visited_all = [True] * self.n
                    self._draw_graph(ax, visited_nodes=visited_all, current_path=comp_path + [comp_path[0]], algorithm=label)
                    frame = self._convert_fig_to_image(fig)
                    lines = [
                        f"{name_map.get(label, label)}：形成一个完整回路", 
                        f"路径：{comp_path + [comp_path[0]]}",
                        f"总长度：{total_cost:.2f}",
                        f"当前最优：{ev['best_cost']:.2f}" if math.isfinite(ev['best_cost']) else "当前最优：∞"
                    ]
                    # Smooth highlight of completion
                    for _ in range(HIGHLIGHT_FRAMES):
                        frames.append(self._draw_text_overlay(frame.copy(), lines, 1.0))
                elif etype == 'update_best':
                    # A new best tour has been found; highlight it strongly
                    bestp = ev['path']
                    total_cost = ev['total_cost']
                    fig, ax = self._init_canvas()
                    self._draw_graph(ax, visited_nodes=[True] * self.n, current_path=bestp + [bestp[0]], algorithm=label)
                    frame = self._convert_fig_to_image(fig)
                    lines = [
                        f"{name_map.get(label, label)}：找到新的最优解", 
                        f"路径：{bestp + [bestp[0]]}",
                        f"总长度：{total_cost:.2f}"
                    ]
                    for _ in range(HIGHLIGHT_FRAMES):
                        frames.append(self._draw_text_overlay(frame.copy(), lines, 1.0))
                # If we exceed a certain number of frames, break early to avoid huge animations
                if len(frames)-countframestart > 1000:
                    break
            # End of events; summarise result for this algorithm
            summary_lines = [
                f"{name_map.get(label, label)} 完成", 
                f"最优回路：{best_path + [best_path[0]]}",
                f"总长度：{best_cost:.2f}",
                f"探索节点数：{expansions}"
            ]
            # Draw summary view with optimal tour
            fig, ax = self._init_canvas()
            self._draw_graph(ax, visited_nodes=[True] * self.n, current_path=best_path + [best_path[0]], algorithm=label)
            frame = self._convert_fig_to_image(fig)
            # Fade in summary text
            for i in range(TEXT_FADE_FRAMES):
                op = (i + 1) / TEXT_FADE_FRAMES
                frames.append(self._draw_text_overlay(frame.copy(), summary_lines, op))
            # Hold summary
            for _ in range(TEXT_HOLD_FRAMES * 2):  # hold a bit longer
                frames.append(self._draw_text_overlay(frame.copy(), summary_lines, 1.0))

        # Animate backtracking
        animate_algorithm(self.bt_events, self.bt_best_path, self.bt_best_cost, self.bt_expansions, 'BT')
        # Animate branch and bound
        animate_algorithm(self.bnb_events, self.bnb_best_path, self.bnb_best_cost, self.bnb_expansions, 'BnB')

        # ------------------------------------------------------------------
        # Phase 3: Final comparison
        # ------------------------------------------------------------------
        comparison_lines = [
            "算法比较", 
            f"回溯法：总长度={self.bt_best_cost:.2f}，探索节点数={self.bt_expansions}",
            f"分支限界法：总长度={self.bnb_best_cost:.2f}，探索节点数={self.bnb_expansions}"
        ]
        fig, ax = self._init_canvas()
        # Draw both optimal tours in the same graph with different colours
        # Use dashed lines for BT and dotted for BnB to distinguish
        # Draw base graph
        self._draw_graph(ax)
        # Draw BT best path
        path_bt = self.bt_best_path + [self.bt_best_path[0]] if self.bt_best_path else []
        for i in range(len(path_bt) - 1):
            a, b = path_bt[i], path_bt[i + 1]
            xi, yi = self.positions[a]
            xj, yj = self.positions[b]
            ax.plot([xi, xj], [yi, yj], color=PATH_COLOUR_BT, linewidth=2.5, linestyle='-', zorder=2)
        # Draw BnB best path
        path_bnb = self.bnb_best_path + [self.bnb_best_path[0]] if self.bnb_best_path else []
        for i in range(len(path_bnb) - 1):
            a, b = path_bnb[i], path_bnb[i + 1]
            xi, yi = self.positions[a]
            xj, yj = self.positions[b]
            ax.plot([xi, xj], [yi, yj], color=PATH_COLOUR_BNB, linewidth=2.5, linestyle='--', zorder=2)
        # Convert to image
        base = self._convert_fig_to_image(fig)
        # Fade in comparison text
        for i in range(TEXT_FADE_FRAMES):
            op = (i + 1) / TEXT_FADE_FRAMES
            frames.append(self._draw_text_overlay(base.copy(), comparison_lines, op))
        # Hold comparison
        for _ in range(TEXT_HOLD_FRAMES * 2):
            frames.append(self._draw_text_overlay(base.copy(), comparison_lines, 1.0))
        return frames

    def save_gif(self, frames: List[Image.Image], filename: str) -> None:
        """Save the frames as a GIF with the configured FPS.  GIF
        compression is applied to reduce the file size while preserving
        visual quality.  This method writes the file to disk."""
        # Convert RGBA images to P mode with adaptive palette for GIF
        # This reduces file size considerably.  We set the palette to 256 colours.
        palette_frames = []
        for img in frames:
            # Convert to P mode while preserving transparency
            # First flatten alpha onto background for consistent palette generation
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            background = Image.new('RGBA', img.size, BACKGROUND_COLOUR + "FF")
            composited = Image.alpha_composite(background, img)
            pal_img = composited.convert('P', palette=Image.ADAPTIVE, colors=256)
            palette_frames.append(pal_img)
        # Duration per frame in milliseconds
        duration_ms = int(1000 / FPS)
        # Save as GIF
        palette_frames[0].save(
            filename,
            save_all=True,
            append_images=palette_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=True
        )


def main():
    # Generate the random graph
    global RANDOM_SEED
    if RANDOM_SEED is None:
        RANDOM_SEED = random.randint(0, 2**32 - 1)
    positions, distances = generate_random_graph(NUM_CITIES, RANDOM_SEED)
    # Solve using backtracking and branch & bound
    bt_path, bt_cost, bt_expansions, bt_events = solve_tsp_backtracking(distances, MAX_EVENTS_PER_ALGO)
    bnb_path, bnb_cost, bnb_expansions, bnb_events = solve_tsp_branch_and_bound(distances, MAX_EVENTS_PER_ALGO)
    # Print results for reproducibility
    print(f"随机种子：{RANDOM_SEED}")
    print(f"回溯法：最优长度 = {bt_cost:.2f}，探索节点数 = {bt_expansions}")
    print(f"分支限界法：最优长度 = {bnb_cost:.2f}，探索节点数 = {bnb_expansions}")
    # Build animation
    anim = TSPAnimation(
        positions=positions,
        distances=distances,
        bt_events=bt_events,
        bnb_events=bnb_events,
        bt_best_path=bt_path,
        bt_best_cost=bt_cost,
        bnb_best_path=bnb_path,
        bnb_best_cost=bnb_cost,
        bt_expansions=bt_expansions,
        bnb_expansions=bnb_expansions
    )
    frames = anim.build_frames()
    # Determine output file name
    out_path = os.path.join(os.path.dirname(__file__), 'tsp_animation.gif')
    anim.save_gif(frames, out_path)
    print(f"Animation saved to {out_path} ({len(frames)} frames)")


if __name__ == '__main__':
    main()