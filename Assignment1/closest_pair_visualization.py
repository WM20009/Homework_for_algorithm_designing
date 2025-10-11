"""
平面最近点对问题 - 分治算法实现与可视化 (超平滑版 - 修复版)
Algorithm Design and Analysis - Closest Pair of Points (Ultra Smooth - Fixed)
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import random
import math
import matplotlib.font_manager as fm



# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ClosestPairVisualizerUltraSmooth:
    def __init__(self, n=50, seed=42):
        """
        初始化超平滑可视化器
        """
        self.n = n
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # 生成满足要求的点集
        self.points = self.generate_points()
        self.frames = []  # 存储动画帧
        self.global_min_distance = float('inf')
        self.global_closest_pair = None
        
        # 平滑动画设置
        self.smooth_frames = 6  # 每个关键操作的平滑帧数
        self.pause_frames = 2   # 关键状态的暂停帧数
        self.highlight_frames = 4  # 高亮效果的帧数
        
    def generate_points(self):
        """生成满足作业要求的点集"""
        mid_line = 0.5
        band_width = 0.1
        points_in_band = int(self.n * 0.35)
        points_left_side = int((self.n - points_in_band) * 0.6)
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
        """暴力法求最近点对"""
        n = len(points)
        if n < 2:
            return float('inf'), None
        
        min_dist = float('inf')
        pair = None
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.distance(points[i], points[j])
                if dist < min_dist:
                    min_dist = dist
                    pair = (i, j)
        
        return min_dist, pair
    
    def strip_closest(self, strip, d, strip_indices):
        """
        在带区内寻找最近点对 (修复版)
        :param strip: 带区内的点
        :param d: 当前最小距离
        :param strip_indices: 带区内点的原始索引
        """
        if len(strip) != len(strip_indices):
            print(f"警告: strip长度({len(strip)}) != strip_indices长度({len(strip_indices)})")
            return d, None
            
        min_dist = d
        pair = None
        
        # 按y坐标排序，同时保持索引对应关系
        strip_sorted = sorted(zip(strip, strip_indices), key=lambda x: x[0][1])
        
        # 对于每个点，检查后续的点
        for i in range(len(strip_sorted)):
            j = i + 1
            while j < len(strip_sorted) and (strip_sorted[j][0][1] - strip_sorted[i][0][1]) < min_dist:
                dist = self.distance(strip_sorted[i][0], strip_sorted[j][0])
                if dist < min_dist:
                    min_dist = dist
                    pair = (strip_sorted[i][1], strip_sorted[j][1])
                j += 1
        
        return min_dist, pair
    
    def update_global_best(self, distance, pair):
        """更新全局最佳结果"""
        if distance < self.global_min_distance:
            old_pair = self.global_closest_pair
            self.global_min_distance = distance
            self.global_closest_pair = pair
            return True, old_pair
        return False, None
    
    def add_smooth_transition(self, transition_type, **kwargs):
        """添加平滑过渡效果"""
        if transition_type == 'divide_line':
            # 分割线滑动动画
            start_x = kwargs.get('start_x', 0)
            end_x = kwargs.get('end_x', 1)
            depth = kwargs.get('depth', 0)
            
            for i in range(self.smooth_frames):
                alpha = i / (self.smooth_frames - 1)
                # 使用缓动函数让动画更自然
                eased_alpha = self.ease_in_out_cubic(alpha)
                current_x = start_x + (end_x - start_x) * eased_alpha
                
                self.frames.append({
                    'type': 'divide_line_smooth',
                    'mid_x': current_x,
                    'alpha': alpha,
                    'depth': depth,
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                })
        
        elif transition_type == 'strip_appear':
            # 带区出现动画
            mid_x = kwargs.get('mid_x')
            delta = kwargs.get('delta')
            strip = kwargs.get('strip', [])
            depth = kwargs.get('depth', 0)
            
            for i in range(self.smooth_frames):
                alpha = i / (self.smooth_frames - 1)
                eased_alpha = self.ease_in_out_cubic(alpha)
                
                self.frames.append({
                    'type': 'strip_appear_smooth',
                    'mid_x': mid_x,
                    'delta': delta,
                    'strip': strip,
                    'alpha': eased_alpha,
                    'depth': depth,
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                })
        
        elif transition_type == 'highlight_pair':
            # 点对高亮动画
            pair = kwargs.get('pair')
            highlight_type = kwargs.get('highlight_type', 'discovery')
            depth = kwargs.get('depth', 0)
            
            for i in range(self.highlight_frames):
                alpha = i / (self.highlight_frames - 1)
                # 使用正弦波创建脉冲效果
                pulse = 0.5 + 0.5 * math.sin(alpha * math.pi * 2)
                
                self.frames.append({
                    'type': 'highlight_pair_smooth',
                    'pair': pair,
                    'highlight_type': highlight_type,
                    'pulse': pulse,
                    'alpha': alpha,
                    'depth': depth,
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                })
    
    def add_pause_frames(self, frame_data, count=None):
        """添加暂停帧"""
        if count is None:
            count = self.pause_frames
        
        for _ in range(count):
            pause_frame = frame_data.copy()
            pause_frame['type'] += '_pause'
            self.frames.append(pause_frame)
    
    @staticmethod
    def ease_in_out_cubic(t):
        """缓动函数：三次方缓入缓出"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def closest_pair_recursive(self, points_x, points_y, indices, depth=0):
        """分治法递归求最近点对 (超平滑版)"""
        n = len(points_x)
        
        # 基本情况：点数<=3时使用暴力法
        if n <= 3:
            # 高亮当前处理的点
            for i in range(self.smooth_frames):
                alpha = i / (self.smooth_frames - 1)
                self.frames.append({
                    'type': 'base_case_highlight',
                    'points': points_x,
                    'indices': indices,
                    'alpha': alpha,
                    'depth': depth,
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                })
            
            min_dist, local_pair = self.brute_force(points_x)
            if local_pair:
                pair = (indices[local_pair[0]], indices[local_pair[1]])
                updated, old_pair = self.update_global_best(min_dist, pair)
                
                # 显示局部结果
                base_frame = {
                    'type': 'base_case',
                    'points': points_x,
                    'indices': indices,
                    'pair': pair,
                    'distance': min_dist,
                    'depth': depth,
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                }
                self.frames.append(base_frame)
                
                # 如果更新了全局最佳，添加高亮动画
                if updated:
                    self.add_smooth_transition('highlight_pair', 
                                             pair=pair, highlight_type='global_update', depth=depth)
                
                # 暂停以便观察
                self.add_pause_frames(base_frame)
            else:
                pair = None
            
            return min_dist, pair
        
        # 分割点
        mid = n // 2
        mid_point = points_x[mid]
        mid_x = mid_point[0]
        
        # 分割线动画：从左边界滑动到分割位置
        self.add_smooth_transition('divide_line', 
                                 start_x=0, end_x=mid_x, depth=depth)
        
        # 显示最终分割状态
        divide_frame = {
            'type': 'divide',
            'mid_x': mid_x,
            'points': points_x,
            'indices': indices,
            'depth': depth,
            'global_min_distance': self.global_min_distance,
            'global_closest_pair': self.global_closest_pair
        }
        self.frames.append(divide_frame)
        self.add_pause_frames(divide_frame)
        
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
        
        # 更新全局最佳
        updated, old_pair = self.update_global_best(d, pair)
        
        # 显示合并结果
        merge_frame = {
            'type': 'merge',
            'mid_x': mid_x,
            'delta': d,
            'pair_left': pair_l,
            'pair_right': pair_r,
            'current_pair': pair,
            'depth': depth,
            'global_min_distance': self.global_min_distance,
            'global_closest_pair': self.global_closest_pair
        }
        self.frames.append(merge_frame)
        
        # 如果更新了全局最佳，添加高亮动画
        if updated:
            self.add_smooth_transition('highlight_pair', 
                                     pair=pair, highlight_type='global_update', depth=depth)
        
        # 构建带区 (修复索引问题)
        strip = []
        strip_indices = []
        for p in points_y:
            if abs(p[0] - mid_x) < d:
                strip.append(p)
                # 找到对应的原始索引
                for idx in indices:
                    if np.allclose(self.points[idx], p, atol=1e-10):
                        strip_indices.append(idx)
                        break
        
        # 确保索引长度匹配
        if len(strip) != len(strip_indices):
            print(f"警告: 带区构建时索引不匹配 strip={len(strip)}, indices={len(strip_indices)}")
            # 重新构建，确保匹配
            strip = []
            strip_indices = []
            for idx in indices:
                p = self.points[idx]
                if abs(p[0] - mid_x) < d:
                    strip.append(p)
                    strip_indices.append(idx)
        
        # 带区出现动画
        self.add_smooth_transition('strip_appear', 
                                 mid_x=mid_x, delta=d, strip=strip, depth=depth)
        
        # 显示带区
        strip_frame = {
            'type': 'strip',
            'mid_x': mid_x,
            'delta': d,
            'strip': strip,
            'strip_indices': strip_indices,
            'depth': depth,
            'global_min_distance': self.global_min_distance,
            'global_closest_pair': self.global_closest_pair
        }
        self.frames.append(strip_frame)
        self.add_pause_frames(strip_frame)
        
        # 在带区中寻找更近的点对
        if len(strip) > 1:
            ds, pair_s = self.strip_closest(strip, d, strip_indices)
            
            if ds < d:
                d = ds
                pair = pair_s
                updated, old_pair = self.update_global_best(d, pair)
                
                # 新发现高亮动画
                self.add_smooth_transition('highlight_pair', 
                                         pair=pair, highlight_type='discovery', depth=depth)
                
                # 显示带区结果
                strip_result_frame = {
                    'type': 'strip_result',
                    'mid_x': mid_x,
                    'delta': d,
                    'pair': pair,
                    'depth': depth,
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                }
                self.frames.append(strip_result_frame)
                
                # 如果更新了全局最佳
                if updated:
                    self.add_smooth_transition('highlight_pair', 
                                             pair=pair, highlight_type='global_update', depth=depth)
                
                self.add_pause_frames(strip_result_frame)
        
        return d, pair
    
    def solve(self):
        """求解最近点对问题"""
        # 初始帧
        initial_frame = {
            'type': 'initial',
            'global_min_distance': self.global_min_distance,
            'global_closest_pair': self.global_closest_pair
        }
        self.frames.append(initial_frame)
        self.add_pause_frames(initial_frame, count=3)
        
        # 排序动画
        indices = list(range(self.n))
        points_x = sorted(zip(self.points, indices), key=lambda x: x[0][0])
        points_x, sorted_indices = zip(*points_x)
        points_x = np.array(points_x)
        sorted_indices = list(sorted_indices)
        
        points_y = sorted(self.points, key=lambda p: p[1])
        
        # 排序过程动画
        for i in range(self.smooth_frames):
            alpha = i / (self.smooth_frames - 1)
            self.frames.append({
                'type': 'sorting_smooth',
                'alpha': alpha,
                'global_min_distance': self.global_min_distance,
                'global_closest_pair': self.global_closest_pair
            })
        
        sorted_frame = {
            'type': 'sorted',
            'global_min_distance': self.global_min_distance,
            'global_closest_pair': self.global_closest_pair
        }
        self.frames.append(sorted_frame)
        self.add_pause_frames(sorted_frame)
        
        # 开始分治算法
        min_dist, pair = self.closest_pair_recursive(
            points_x, points_y, sorted_indices
        )
        
        # 最终结果高亮动画
        for i in range(self.highlight_frames * 2):
            alpha = i / (self.highlight_frames * 2 - 1)
            pulse = 0.7 + 0.3 * math.sin(alpha * math.pi * 4)
            
            self.frames.append({
                'type': 'final_celebration',
                'pair': self.global_closest_pair,
                'distance': self.global_min_distance,
                'pulse': pulse,
                'alpha': alpha,
                'global_min_distance': self.global_min_distance,
                'global_closest_pair': self.global_closest_pair
            })
        
        # 最终静态帧
        final_frame = {
            'type': 'final',
            'pair': self.global_closest_pair,
            'distance': self.global_min_distance,
            'global_min_distance': self.global_min_distance,
            'global_closest_pair': self.global_closest_pair
        }
        self.frames.append(final_frame)
        self.add_pause_frames(final_frame, count=5)
        
        return self.global_min_distance, self.global_closest_pair
    
    def create_animation(self, output_file='closest_pair_ultra_smooth.gif', fps=8, use_pillow=True):
        """创建超平滑动画"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        def update(frame_idx):
            ax.clear()
            frame = self.frames[frame_idx]
            frame_type = frame['type']
            
            # 获取全局状态
            global_min_dist = frame.get('global_min_distance', float('inf'))
            global_pair = frame.get('global_closest_pair', None)
            
            # 绘制所有点（基础层）
            base_color = 'lightblue'
            base_alpha = 0.6
            
            # 根据帧类型调整基础点的显示
            if 'highlight' in frame_type or 'smooth' in frame_type:
                base_alpha = 0.3
            
            ax.scatter(self.points[:, 0], self.points[:, 1], 
                      c=base_color, s=50, alpha=base_alpha, 
                      edgecolors='black', linewidth=0.5, zorder=1)
            
            # 绘制全局最佳点对（永远在顶层）
            if global_pair and global_min_dist != float('inf'):
                p1, p2 = self.points[global_pair[0]], self.points[global_pair[1]]
                
                global_alpha = 0.8
                global_size = 120
                global_linewidth = 3
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       'gold', linewidth=global_linewidth, alpha=global_alpha, 
                       label=f"全局最佳 {global_min_dist:.4f}", zorder=8)
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                          c='gold', s=global_size, edgecolors='orange', 
                          linewidth=2, zorder=9, alpha=global_alpha)
            
            # 处理不同类型的帧
            title = f"帧 {frame_idx + 1}/{len(self.frames)}: "
            info_text = ""
            
            if frame_type in ['initial', 'initial_pause']:
                title += "初始点集"
                info_text = f"总点数: {self.n}\n随机种子: {self.seed}"
                
            elif frame_type == 'sorting_smooth':
                alpha = frame.get('alpha', 0)
                title += f"排序中... ({alpha*100:.0f}%)"
                info_text = "按X坐标排序"
                
            elif frame_type in ['sorted', 'sorted_pause']:
                title += "排序完成"
                info_text = "预处理完成\n开始分治算法"
                
            elif frame_type == 'divide_line_smooth':
                alpha = frame.get('alpha', 0)
                mid_x = frame.get('mid_x', 0.5)
                depth = frame.get('depth', 0)
                title += f"分割线移动中 (深度 {depth})"
                
                line_alpha = 0.3 + 0.7 * alpha
                ax.axvline(x=mid_x, color='red', linestyle='--', 
                          linewidth=2, alpha=line_alpha, label='分割线')
                info_text = f"分割线 x = {mid_x:.3f}"
                
            elif frame_type in ['divide', 'divide_pause']:
                mid_x = frame.get('mid_x', 0.5)
                depth = frame.get('depth', 0)
                title += f"分治 - 分割 (深度 {depth})"
                ax.axvline(x=mid_x, color='red', linestyle='--', linewidth=2, label='分割线')
                info_text = f"分割线 x = {mid_x:.3f}"
                
            elif frame_type == 'base_case_highlight':
                alpha = frame.get('alpha', 0)
                depth = frame.get('depth', 0)
                points = frame.get('points', [])
                title += f"基本情况处理中 (深度 {depth})"
                
                highlight_alpha = 0.3 + 0.7 * math.sin(alpha * math.pi)
                if len(points) > 0:
                    ax.scatter(points[:, 0], points[:, 1], 
                              c='orange', s=80 + 40*highlight_alpha, 
                              edgecolors='black', linewidth=1.5, 
                              zorder=5, alpha=highlight_alpha)
                
            elif frame_type in ['base_case', 'base_case_pause']:
                depth = frame.get('depth', 0)
                points = frame.get('points', [])
                pair = frame.get('pair')
                title += f"基本情况 (深度 {depth})"
                
                if len(points) > 0:
                    ax.scatter(points[:, 0], points[:, 1], 
                              c='orange', s=100, edgecolors='black', 
                              linewidth=1.5, zorder=5)
                
                if pair:
                    p1, p2 = self.points[pair[0]], self.points[pair[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'g-', linewidth=2, label=f"局部最近点对")
                    info_text = f"局部距离 = {frame['distance']:.4f}"
                
            elif frame_type in ['merge', 'merge_pause']:
                depth = frame.get('depth', 0)
                mid_x = frame.get('mid_x', 0.5)
                delta = frame.get('delta', 0)
                current_pair = frame.get('current_pair')
                title += f"合并阶段 (深度 {depth})"
                
                ax.axvline(x=mid_x, color='red', linestyle='--', linewidth=2, label='分割线')
                
                if current_pair:
                    p1, p2 = self.points[current_pair[0]], self.points[current_pair[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'purple', linewidth=2, label=f"当前最近点对")
                
                info_text = f"当前δ = {delta:.4f}"
                
            elif frame_type == 'strip_appear_smooth':
                alpha = frame.get('alpha', 0)
                depth = frame.get('depth', 0)
                mid_x = frame.get('mid_x', 0.5)
                delta = frame.get('delta', 0)
                strip = frame.get('strip', [])
                title += f"带区出现中 (深度 {depth})"
                
                rect_alpha = 0.1 + 0.1 * alpha
                rect = Rectangle((mid_x - delta, 0), 2 * delta, 1, 
                                alpha=rect_alpha, facecolor='yellow', 
                                edgecolor='orange', linewidth=2)
                ax.add_patch(rect)
                ax.axvline(x=mid_x, color='red', linestyle='--', linewidth=2)
                
                if len(strip) > 0:
                    strip_array = np.array(strip)
                    point_alpha = alpha
                    ax.scatter(strip_array[:, 0], strip_array[:, 1], 
                              c='yellow', s=60 + 40*alpha, 
                              edgecolors='orange', linewidth=2, 
                              zorder=5, alpha=point_alpha)
                
            elif frame_type in ['strip', 'strip_pause']:
                depth = frame.get('depth', 0)
                mid_x = frame.get('mid_x', 0.5)
                delta = frame.get('delta', 0)
                strip = frame.get('strip', [])
                title += f"检查带区 (深度 {depth})"
                
                rect = Rectangle((mid_x - delta, 0), 2 * delta, 1, 
                                alpha=0.2, facecolor='yellow', 
                                edgecolor='orange', linewidth=2)
                ax.add_patch(rect)
                ax.axvline(x=mid_x, color='red', linestyle='--', linewidth=2, label='分割线')
                
                if len(strip) > 0:
                    strip_array = np.array(strip)
                    ax.scatter(strip_array[:, 0], strip_array[:, 1], 
                              c='yellow', s=100, edgecolors='orange', 
                              linewidth=2, zorder=5)
                
                info_text = f"δ = {delta:.4f}\n带区宽度 = {2*delta:.4f}\n带区点数 = {len(strip)}"
                
            elif frame_type == 'highlight_pair_smooth':
                depth = frame.get('depth', 0)
                pair = frame.get('pair')
                pulse = frame.get('pulse', 1)
                highlight_type = frame.get('highlight_type', 'discovery')
                
                if highlight_type == 'discovery':
                    title += f"发现新的最近点对! (深度 {depth})"
                else:
                    title += f"更新全局最佳! (深度 {depth})"
                
                if pair:
                    p1, p2 = self.points[pair[0]], self.points[pair[1]]
                    line_width = 2 + 2 * pulse
                    point_size = 100 + 50 * pulse
                    color = 'red' if highlight_type == 'discovery' else 'gold'
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color, linewidth=line_width, label=f"新最近点对", zorder=6)
                    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                              c=color, s=point_size, edgecolors='darkred', 
                              linewidth=2, zorder=7, alpha=pulse)
                
            elif frame_type in ['strip_result', 'strip_result_pause']:
                depth = frame.get('depth', 0)
                mid_x = frame.get('mid_x', 0.5)
                delta = frame.get('delta', 0)
                pair = frame.get('pair')
                title += f"带区发现更近点对! (深度 {depth})"
                
                rect = Rectangle((mid_x - delta, 0), 2 * delta, 1, 
                                alpha=0.2, facecolor='yellow', 
                                edgecolor='orange', linewidth=2)
                ax.add_patch(rect)
                ax.axvline(x=mid_x, color='red', linestyle='--', linewidth=2)
                
                if pair:
                    p1, p2 = self.points[pair[0]], self.points[pair[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'red', linewidth=3, label=f"新最近点对", zorder=6)
                    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                              c='red', s=150, edgecolors='darkred', 
                              linewidth=2, zorder=7)
                
                info_text = f"新距离 = {delta:.4f}"
                
            elif frame_type == 'final_celebration':
                alpha = frame.get('alpha', 0)
                pulse = frame.get('pulse', 1)
                pair = frame.get('pair')
                distance = frame.get('distance', 0)
                title += " 算法完成! "
                
                if pair:
                    p1, p2 = self.points[pair[0]], self.points[pair[1]]
                    
                    # 庆祝动画：多重圆圈效果
                    for i in range(3):
                        radius = (distance/2) * (1 + i * 0.5) * pulse
                        circle = Circle(p1, radius, fill=False, 
                                      edgecolor='red', linestyle=':', 
                                      linewidth=2-i*0.5, alpha=0.5-i*0.1)
                        ax.add_patch(circle)
                        circle = Circle(p2, radius, fill=False, 
                                      edgecolor='red', linestyle=':', 
                                      linewidth=2-i*0.5, alpha=0.5-i*0.1)
                        ax.add_patch(circle)
                    
                    line_width = 3 + 2 * pulse
                    point_size = 150 + 100 * pulse
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'red', linewidth=line_width, label=f"最近点对", zorder=6)
                    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                              c='red', s=point_size, edgecolors='darkred', 
                              linewidth=3, zorder=7)
                
                info_text = f"最小距离 = {distance:.6f}"
                
            elif frame_type in ['final', 'final_pause']:
                title += "最终结果"
                pair = frame.get('pair')
                distance = frame.get('distance', 0)
                
                if pair:
                    p1, p2 = self.points[pair[0]], self.points[pair[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'red', linewidth=4, label=f"最近点对", zorder=6)
                    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                              c='red', s=200, edgecolors='darkred', 
                              linewidth=2, zorder=7)
                    
                    # 距离圆圈
                    circle1 = Circle(p1, distance/2, fill=False, 
                                   edgecolor='red', linestyle=':', 
                                   linewidth=1.5, alpha=0.5)
                    circle2 = Circle(p2, distance/2, fill=False, 
                                   edgecolor='red', linestyle=':', 
                                   linewidth=1.5, alpha=0.5)
                    ax.add_patch(circle1)
                    ax.add_patch(circle2)
                
                    info_text = f"最小距离 = {distance:.6f}\n"
                    info_text += f"点对: ({p1[0]:.3f}, {p1[1]:.3f}) ↔ ({p2[0]:.3f}, {p2[1]:.3f})"
            
            # 设置坐标轴
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.set_xlabel('X 坐标', fontsize=12)
            ax.set_ylabel('Y 坐标', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 左下角信息文本框
            if info_text:
                ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 右上角全局状态面板
            status_text = "全局状态\n"
            if global_min_dist != float('inf'):
                status_text += f"最短距离: {global_min_dist:.6f}\n"
                if global_pair:
                    p1, p2 = self.points[global_pair[0]], self.points[global_pair[1]]
                    status_text += f"点对: P{global_pair[0]} ↔ P{global_pair[1]}\n"
                    status_text += f"坐标: ({p1[0]:.3f}, {p1[1]:.3f})\n"
                    status_text += f"     ({p2[0]:.3f}, {p2[1]:.3f})"
            else:
                status_text += "暂无结果"
            
            status_color = 'lightgreen' if global_pair else 'lightgray'
            ax.text(0.98, 0.98, status_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.9))
        
        anim = animation.FuncAnimation(fig, update, frames=len(self.frames),
                                      interval=1000//fps, repeat=True)
        
        # 保存动画
        if use_pillow or output_file.endswith('.gif'):
            Writer = animation.writers['pillow']
            writer = Writer(fps=fps)
            anim.save(output_file, writer=writer, dpi=120)
        else:
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Algorithm Visualization'), bitrate=1800)
                anim.save(output_file, writer=writer, dpi=120)
            except RuntimeError:
                Writer = animation.writers['pillow']
                writer = Writer(fps=fps)
                output_file = output_file.replace('.mp4', '.gif')
                anim.save(output_file, writer=writer, dpi=120)
        
        plt.close()
        print(f"超平滑动画已保存到: {output_file}")
        print(f"总帧数: {len(self.frames)}")
 
 
def main():
    """主函数"""
    N = 20  # 减少点数让动画更清晰，避免复杂的索引问题
    SEED = 42
    OUTPUT_FILE = 'closest_pair_ultra_smooth_fixed.gif'
    FPS = 6  # 稍微降低帧率
    
    print("=" * 60)
    print("平面最近点对问题 - 分治算法可视化 (超平滑版-修复)")
    print("=" * 60)
    print(f"点数量: {N}")
    print(f"随机种子: {SEED}")
    print()
    
    # 创建超平滑可视化器
    visualizer = ClosestPairVisualizerUltraSmooth(n=N, seed=SEED)
    min_dist, pair = visualizer.solve()
    
    print("算法求解完成！")
    print(f"最小距离: {min_dist:.6f}")
    if pair:
        p1, p2 = visualizer.points[pair[0]], visualizer.points[pair[1]]
        print(f"最近点对:")
        print(f"  点1: ({p1[0]:.6f}, {p1[1]:.6f})")
        print(f"  点2: ({p2[0]:.6f}, {p2[1]:.6f})")
    
    print(f"\n准备生成 {len(visualizer.frames)} 帧的超平滑动画...")
    print("正在生成超平滑动画...")
    visualizer.create_animation(output_file=OUTPUT_FILE, fps=FPS)
    print()
    print("=" * 60)
    print("完成！")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()