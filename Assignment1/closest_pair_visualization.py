"""
平面最近点对问题 - 分治算法实现与可视化
Algorithm Design and Analysis - Closest Pair of Points
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ClosestPairVisualizer:
    def __init__(self, n=50, seed=42):
        """
        初始化可视化器
        :param n: 点的数量
        :param seed: 随机种子
        """
        self.n = n
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # 生成满足要求的点集
        self.points = self.generate_points()
        self.frames = []  # 存储动画帧
        self.min_distance = float('inf')
        self.closest_pair = None
        
    def generate_points(self):
        """
        生成满足作业要求的点集：
        1. 点位于分界线(x=0.5)两侧
        2. 左右两侧点数相差10%-20%
        3. 在中线±10%宽度内至少有30%的点
        """
        mid_line = 0.5
        band_width = 0.1  # ±10%
        points_in_band = int(self.n * 0.35)  # 35%的点在带区
        points_left_side = int((self.n - points_in_band) * 0.6)  # 左侧60%
        points_right_side = (self.n - points_in_band) - points_left_side
        
        points = []
        
        # 生成带区内的点
        for _ in range(points_in_band):
            x = np.random.uniform(mid_line - band_width, mid_line + band_width)
            y = np.random.uniform(0, 1)
            points.append([x, y])
        
        # 生成左侧的点
        for _ in range(points_left_side):
            x = np.random.uniform(0, mid_line - band_width)
            y = np.random.uniform(0, 1)
            points.append([x, y])
        
        # 生成右侧的点
        for _ in range(points_right_side):
            x = np.random.uniform(mid_line + band_width, 1)
            y = np.random.uniform(0, 1)
            points.append([x, y])
        
        return np.array(points)
    
    def distance(self, p1, p2):
        """计算两点间的欧氏距离"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def brute_force(self, points):
        """暴力法求最近点对，用于点数<=3的情况"""
        min_dist = float('inf')
        pair = None
        n = len(points)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(points[i], points[j])
                if d < min_dist:
                    min_dist = d
                    pair = (i, j)
        
        return min_dist, pair
    
    def strip_closest(self, strip, d, strip_indices):
        """
        在带区内寻找最近点对
        :param strip: 带区内的点
        :param d: 当前最小距离
        :param strip_indices: 带区内点的原始索引
        """
        min_dist = d
        pair = None
        
        # 按y坐标排序
        strip_sorted = sorted(zip(strip, strip_indices), key=lambda x: x[0][1])
        
        # 对于每个点，只需检查后续最多6个点
        for i in range(len(strip_sorted)):
            j = i + 1
            while j < len(strip_sorted) and (strip_sorted[j][0][1] - strip_sorted[i][0][1]) < min_dist:
                d = self.distance(strip_sorted[i][0], strip_sorted[j][0])
                if d < min_dist:
                    min_dist = d
                    pair = (strip_sorted[i][1], strip_sorted[j][1])
                j += 1
        
        return min_dist, pair
    
    def closest_pair_recursive(self, points_x, points_y, indices, depth=0):
        """
        分治法递归求最近点对
        :param points_x: 按x坐标排序的点
        :param points_y: 按y坐标排序的点
        :param indices: 点的原始索引
        :param depth: 递归深度
        """
        n = len(points_x)
        
        # 基本情况：点数<=3时使用暴力法
        if n <= 3:
            min_dist, local_pair = self.brute_force(points_x)
            if local_pair:
                pair = (indices[local_pair[0]], indices[local_pair[1]])
            else:
                pair = None
            
            # 记录帧：基本情况
            self.frames.append({
                'type': 'base_case',
                'points': points_x,
                'indices': indices,
                'pair': pair,
                'distance': min_dist,
                'depth': depth
            })
            
            return min_dist, pair
        
        # 分割点
        mid = n // 2
        mid_point = points_x[mid]
        mid_x = mid_point[0]
        
        # 记录帧：显示分割线
        self.frames.append({
            'type': 'divide',
            'mid_x': mid_x,
            'points': points_x,
            'indices': indices,
            'depth': depth
        })
        
        # 分割点集
        points_y_left = [p for p in points_y if p[0] <= mid_x]
        points_y_right = [p for p in points_y if p[0] > mid_x]
        
        indices_left = indices[:mid]
        indices_right = indices[mid:]
        
        # 递归求解左右两侧
        dl, pair_l = self.closest_pair_recursive(
            points_x[:mid], points_y_left, indices_left, depth + 1
        )
        dr, pair_r = self.closest_pair_recursive(
            points_x[mid:], points_y_right, indices_right, depth + 1
        )
        
        # 取较小的距离
        if dl < dr:
            d = dl
            pair = pair_l
        else:
            d = dr
            pair = pair_r
        
        # 记录帧：显示左右最小距离
        self.frames.append({
            'type': 'merge',
            'mid_x': mid_x,
            'delta': d,
            'pair_left': pair_l,
            'pair_right': pair_r,
            'current_pair': pair,
            'depth': depth
        })
        
        # 构建带区
        strip = []
        strip_indices = []
        for i, p in enumerate(points_y):
            if abs(p[0] - mid_x) < d:
                strip.append(p)
                # 找到原始索引
                for j, idx in enumerate(indices):
                    if np.array_equal(self.points[idx], p):
                        strip_indices.append(idx)
                        break
        
        # 记录帧：显示带区
        self.frames.append({
            'type': 'strip',
            'mid_x': mid_x,
            'delta': d,
            'strip': strip,
            'strip_indices': strip_indices,
            'depth': depth
        })
        
        # 在带区中寻找更近的点对
        if len(strip) > 1:
            ds, pair_s = self.strip_closest(strip, d, strip_indices)
            
            if ds < d:
                d = ds
                pair = pair_s
                
                # 记录帧：带区找到更近的点对
                self.frames.append({
                    'type': 'strip_result',
                    'mid_x': mid_x,
                    'delta': d,
                    'pair': pair,
                    'depth': depth
                })
        
        return d, pair
    
    def solve(self):
        """求解最近点对问题"""
        # 按x和y坐标分别排序
        indices = list(range(len(self.points)))
        points_x = sorted(zip(self.points, indices), key=lambda x: x[0][0])
        points_y = sorted(zip(self.points, indices), key=lambda x: x[0][1])
        
        points_x_sorted = np.array([p[0] for p in points_x])
        indices_sorted = [p[1] for p in points_x]
        points_y_sorted = np.array([p[0] for p in points_y])
        
        # 记录帧：初始状态
        self.frames.append({
            'type': 'initial',
            'points': self.points
        })
        
        # 记录帧：排序后状态
        self.frames.append({
            'type': 'sorted',
            'points': points_x_sorted
        })
        
        # 递归求解
        self.min_distance, self.closest_pair = self.closest_pair_recursive(
            points_x_sorted, points_y_sorted, indices_sorted
        )
        
        # 记录帧：最终结果
        self.frames.append({
            'type': 'final',
            'pair': self.closest_pair,
            'distance': self.min_distance
        })
        
        return self.min_distance, self.closest_pair
    
    def create_animation(self, output_file='closest_pair.gif', fps=2, use_pillow=True):
        """
        创建动画
        :param output_file: 输出文件名（.gif 或 .mp4）
        :param fps: 帧率
        :param use_pillow: 使用 Pillow 保存 GIF（无需 ffmpeg）
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def update(frame_idx):
            ax.clear()
            frame = self.frames[frame_idx]
            frame_type = frame['type']
            
            # 绘制所有点
            ax.scatter(self.points[:, 0], self.points[:, 1], 
                      c='lightblue', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # 设置标题和说明
            title = f"帧 {frame_idx + 1}/{len(self.frames)}: "
            info_text = ""
            
            if frame_type == 'initial':
                title += "初始点集"
                info_text = f"总点数: {self.n}\n随机种子: {self.seed}"
                
            elif frame_type == 'sorted':
                title += "按X坐标排序"
                info_text = "预处理完成"
                
            elif frame_type == 'divide':
                title += f"分治 - 分割 (深度 {frame['depth']})"
                mid_x = frame['mid_x']
                ax.axvline(x=mid_x, color='red', linestyle='--', linewidth=2, label='分割线')
                info_text = f"分割线 x = {mid_x:.3f}"
                
            elif frame_type == 'base_case':
                title += f"基本情况 (深度 {frame['depth']})"
                points = frame['points']
                indices = frame['indices']
                ax.scatter(points[:, 0], points[:, 1], 
                          c='orange', s=100, edgecolors='black', linewidth=1.5, zorder=5)
                if frame['pair']:
                    p1_idx, p2_idx = frame['pair']
                    p1, p2 = self.points[p1_idx], self.points[p2_idx]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'g-', linewidth=2, label=f"最近点对")
                    info_text = f"距离 = {frame['distance']:.4f}"
                
            elif frame_type == 'merge':
                title += f"合并阶段 (深度 {frame['depth']})"
                mid_x = frame['mid_x']
                delta = frame['delta']
                ax.axvline(x=mid_x, color='red', linestyle='--', linewidth=2, label='分割线')
                
                if frame['current_pair']:
                    p1_idx, p2_idx = frame['current_pair']
                    p1, p2 = self.points[p1_idx], self.points[p2_idx]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'purple', linewidth=2, label=f"当前最近点对")
                
                info_text = f"δ = {delta:.4f}"
                
            elif frame_type == 'strip':
                title += f"检查带区 (深度 {frame['depth']})"
                mid_x = frame['mid_x']
                delta = frame['delta']
                
                # 绘制带区
                rect = Rectangle((mid_x - delta, 0), 2 * delta, 1, 
                                alpha=0.2, facecolor='yellow', edgecolor='orange', linewidth=2)
                ax.add_patch(rect)
                ax.axvline(x=mid_x, color='red', linestyle='--', linewidth=2, label='分割线')
                
                # 高亮带区内的点
                strip = frame['strip']
                if len(strip) > 0:
                    strip_array = np.array(strip)
                    ax.scatter(strip_array[:, 0], strip_array[:, 1], 
                              c='yellow', s=100, edgecolors='orange', linewidth=2, zorder=5)
                
                info_text = f"δ = {delta:.4f}\n带区宽度 = {2*delta:.4f}\n带区点数 = {len(strip)}"
                
            elif frame_type == 'strip_result':
                title += f"带区发现更近点对! (深度 {frame['depth']})"
                mid_x = frame['mid_x']
                delta = frame['delta']
                
                rect = Rectangle((mid_x - delta, 0), 2 * delta, 1, 
                                alpha=0.2, facecolor='yellow', edgecolor='orange', linewidth=2)
                ax.add_patch(rect)
                ax.axvline(x=mid_x, color='red', linestyle='--', linewidth=2)
                
                if frame['pair']:
                    p1_idx, p2_idx = frame['pair']
                    p1, p2 = self.points[p1_idx], self.points[p2_idx]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'red', linewidth=3, label=f"新最近点对", zorder=6)
                    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                              c='red', s=150, edgecolors='darkred', linewidth=2, zorder=7)
                
                info_text = f"新距离 = {delta:.4f}"
                
            elif frame_type == 'final':
                title += "最终结果"
                if frame['pair']:
                    p1_idx, p2_idx = frame['pair']
                    p1, p2 = self.points[p1_idx], self.points[p2_idx]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'red', linewidth=3, label=f"最近点对", zorder=6)
                    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                              c='red', s=200, edgecolors='darkred', linewidth=2, zorder=7)
                    
                    # 绘制圆圈表示最小距离
                    circle1 = Circle(p1, frame['distance']/2, fill=False, 
                                   edgecolor='red', linestyle=':', linewidth=1.5, alpha=0.5)
                    circle2 = Circle(p2, frame['distance']/2, fill=False, 
                                   edgecolor='red', linestyle=':', linewidth=1.5, alpha=0.5)
                    ax.add_patch(circle1)
                    ax.add_patch(circle2)
                
                info_text = f"最小距离 = {frame['distance']:.6f}\n"
                info_text += f"点对: ({p1[0]:.3f}, {p1[1]:.3f}) ↔ ({p2[0]:.3f}, {p2[1]:.3f})"
            
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.set_xlabel('X 坐标', fontsize=12)
            ax.set_ylabel('Y 坐标', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 添加信息文本框
            if info_text:
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        anim = animation.FuncAnimation(fig, update, frames=len(self.frames),
                                      interval=1000//fps, repeat=True)
        
        # 保存动画
        # 保存动画
        if use_pillow or output_file.endswith('.gif'):
            Writer = animation.writers['pillow']
            writer = Writer(fps=fps)
            anim.save(output_file, writer=writer, dpi=120)
        else:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Algorithm Visualization'),
                        bitrate=1800)
            anim.save(output_file, writer=writer, dpi=120)
        plt.close()
        
        print(f"动画已保存到: {output_file}")


def main():
    """主函数"""
    # 参数设置
    N = 50  # 点的数量 (>=36)
    SEED = 42  # 随机种子
    OUTPUT_FILE = 'closest_pair_animation.gif'
    FPS = 2  # 帧率
    
    print("=" * 60)
    print("平面最近点对问题 - 分治算法可视化")
    print("=" * 60)
    print(f"点数量: {N}")
    print(f"随机种子: {SEED}")
    print()
    
    # 创建可视化器并求解
    visualizer = ClosestPairVisualizer(n=N, seed=SEED)
    min_dist, pair = visualizer.solve()
    
    # 输出结果
    print("算法求解完成！")
    print(f"最小距离: {min_dist:.6f}")
    if pair:
        p1, p2 = visualizer.points[pair[0]], visualizer.points[pair[1]]
        print(f"最近点对:")
        print(f"  点1: ({p1[0]:.6f}, {p1[1]:.6f})")
        print(f"  点2: ({p2[0]:.6f}, {p2[1]:.6f})")
    print()
    
    # 生成动画
    print("正在生成动画...")
    visualizer.create_animation(output_file=OUTPUT_FILE, fps=FPS)
    print()
    print("=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()