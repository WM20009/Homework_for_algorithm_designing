# 美化图例 - 修复：只在有标签时显示图例，并添加淡入淡出
# -*- coding: utf-8 -*-
"""
平面最近点对问题 - 分治算法实现与可视化 (超平滑优化版)
Algorithm Design and Analysis - Closest Pair of Points (Ultra Smooth Version)
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import random
import math

# ============= 美化配置 =============
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
 
# 现代配色方案
class ModernColors:
    PRIMARY = '#2E3440'
    SECONDARY = '#3B4252'
    ACCENT = '#5E81AC'
    
    BG_MAIN = '#ECEFF4'
    BG_CARD = '#FFFFFF'
    
    POINT_NORMAL = '#81A1C1'
    POINT_HIGHLIGHT = '#EBCB8B'
    POINT_ACTIVE = '#BF616A'
    POINT_BEST = '#A3BE8C'
    
    LINE_DIVIDE = '#D08770'
    LINE_BEST = '#A3BE8C'
    LINE_CURRENT = '#B48EAD'
    LINE_NEW = '#BF616A'
    
    STRIP_FILL = '#EBCB8B'
    STRIP_BORDER = '#D08770'
    
    TEXT_PRIMARY = '#2E3440'
    TEXT_SECONDARY = '#4C566A'
    TEXT_LIGHT = '#D8DEE9'
    
    SUCCESS_BOX = '#A3BE8C'
    INFO_BOX = '#81A1C1'
    WARNING_BOX = '#EBCB8B'
    ERROR_BOX = '#BF616A'
 
class Symbols:
    TARGET = "◉"
    ARROW = "→"
    STAR = "★"
    CIRCLE = "○"
    DIAMOND = "◆"
    CHECK = "√"
    SEARCH = "Search"
    INFO = "i"
 
plt.style.use('default')
plt.rcParams['figure.facecolor'] = ModernColors.BG_MAIN
plt.rcParams['axes.facecolor'] = ModernColors.BG_CARD
plt.rcParams['axes.edgecolor'] = ModernColors.SECONDARY
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = ModernColors.SECONDARY
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['text.color'] = ModernColors.TEXT_PRIMARY
 
class ClosestPairVisualizerSmooth:
    def __init__(self, n=20, seed=42):
        """初始化超平滑版可视化器"""
        self.n = n
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.points = self.generate_points()
        self.frames = []
        self.global_min_distance = float('inf')
        self.global_closest_pair = None
        
        # 增加过渡帧数量，提升流畅度
        self.smooth_frames = 15  # 增加到15帧，条带更平滑
        self.pause_frames = 6    # 增加到6帧，让人看清
        self.highlight_frames = 10  # 增加到10帧
        self.text_fade_frames = 6  # 增加到6帧
        self.text_hold_frames = 8  # 增加到8帧，图例显示更久
        self.legend_fade_frames = 8  # 新增：图例淡入淡出帧数

        # 可单独调整关键帧停留时长（以 pause_frames 为基准）
        self.divide_pause_frames = max(1, self.pause_frames * 2)   # 分割帧延长为原来的 2 倍
        self.strip_pause_frames = max(1, self.pause_frames * 2)    # 带区帧延长为原来的 2 倍
        
    def generate_points(self):
        """生成满足作业要求的点集"""
        mid_line = 0.5
        band_width = 0.1
        points_in_band = int(self.n * 0.35)
        points_left_side = int((self.n - points_in_band) * 0.6)
        points_right_side = (self.n - points_in_band) - points_left_side
        
        points = []
        
        for _ in range(points_in_band):
            x = np.random.uniform(mid_line - band_width, mid_line + band_width)
            y = np.random.uniform(0, 1)
            points.append([x, y])
        
        for _ in range(points_left_side):
            x = np.random.uniform(0, mid_line - band_width)
            y = np.random.uniform(0, 1)
            points.append([x, y])
        
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
        """在带区内寻找最近点对"""
        if len(strip) != len(strip_indices):
            return d, None
            
        min_dist = d
        pair = None
        
        strip_sorted = sorted(zip(strip, strip_indices), key=lambda x: x[0][1])
        
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
            start_x = kwargs.get('start_x', 0)
            end_x = kwargs.get('end_x', 1)
            depth = kwargs.get('depth', 0)
            
            for i in range(self.smooth_frames):
                alpha = i / (self.smooth_frames - 1)
                eased_alpha = self.ease_in_out_cubic(alpha)
                current_x = start_x + (end_x - start_x) * eased_alpha
                
                self.frames.append({
                    'type': 'divide_line_smooth',
                    'mid_x': current_x,
                    'alpha': alpha,
                    'depth': depth,
                    'legend_alpha': min(1.0, alpha * 2),  # 图例淡入
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                })
        
        elif transition_type == 'strip_appear':
            mid_x = kwargs.get('mid_x')
            delta = kwargs.get('delta')
            strip = kwargs.get('strip', [])
            strip_indices = kwargs.get('strip_indices', [])
            depth = kwargs.get('depth', 0)
            
            # 条带宽度淡入效果（从0到delta）
            for i in range(self.smooth_frames):
                alpha = i / (self.smooth_frames - 1)
                eased_alpha = self.ease_in_out_cubic(alpha)
                current_delta = delta * eased_alpha  # 条带宽度逐渐增加
                
                self.frames.append({
                    'type': 'strip_appear_smooth',
                    'mid_x': mid_x,
                    'delta': current_delta,  # 使用渐变的宽度
                    'target_delta': delta,   # 保存目标宽度
                    'strip': strip,
                    'strip_indices': strip_indices,
                    'alpha': eased_alpha,
                    'legend_alpha': min(1.0, alpha * 1.5),  # 图例淡入
                    'depth': depth,
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                })
        
        elif transition_type == 'highlight_pair':
            pair = kwargs.get('pair')
            highlight_type = kwargs.get('highlight_type', 'discovery')
            depth = kwargs.get('depth', 0)
            
            for i in range(self.highlight_frames):
                alpha = i / (self.highlight_frames - 1)
                pulse = 0.5 + 0.5 * math.sin(alpha * math.pi * 2)
                
                self.frames.append({
                    'type': 'highlight_pair_smooth',
                    'pair': pair,
                    'highlight_type': highlight_type,
                    'pulse': pulse,
                    'alpha': alpha,
                    'legend_alpha': 1.0,  # 高亮时图例完全显示
                    'depth': depth,
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                })
    
    def add_text_transition(self, text_type, **kwargs):
        """添加文字平滑过渡效果"""
        text_content = kwargs.get('text_content', '')
        frame_base = kwargs.get('frame_base', {})
        
        for i in range(self.text_fade_frames):
            alpha = i / (self.text_fade_frames - 1)
            fade_frame = frame_base.copy()
            fade_frame['type'] = f'{fade_frame["type"]}_text_fade_in'
            fade_frame['text_alpha'] = alpha
            fade_frame['text_content'] = text_content
            fade_frame['text_type'] = text_type
            self.frames.append(fade_frame)
        
        for i in range(self.text_hold_frames):
            hold_frame = frame_base.copy()
            hold_frame['type'] = f'{hold_frame["type"]}_text_hold'
            hold_frame['text_alpha'] = 1.0
            hold_frame['text_content'] = text_content
            hold_frame['text_type'] = text_type
            self.frames.append(hold_frame)
        
        for i in range(self.text_fade_frames):
            alpha = 1.0 - (i / (self.text_fade_frames - 1))
            fade_frame = frame_base.copy()
            fade_frame['type'] = f'{fade_frame["type"]}_text_fade_out'
            fade_frame['text_alpha'] = alpha
            fade_frame['text_content'] = text_content
            fade_frame['text_type'] = text_type
            self.frames.append(fade_frame)
    
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
        """分治法递归求最近点对"""
        n = len(points_x)
        
        if n <= 3:
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
                
                if updated:
                    self.add_text_transition('global_update', 
                                            text_content=f"发现新的全局最佳距离: {min_dist:.4f}!",
                                            frame_base=base_frame)
                
                # 延长基本情况帧停留
                self.add_pause_frames(base_frame, count=max(1, self.pause_frames * 2))
            else:
                pair = None
            
            return min_dist, pair
        
        mid = n // 2
        mid_point = points_x[mid]
        mid_x = mid_point[0]
        
        self.add_smooth_transition('divide_line', 
                                 start_x=0, end_x=mid_x, depth=depth)
        
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
        
        if depth == 0:
            self.add_text_transition('algorithm_start',
                                    text_content="开始分治算法求解!",
                                    frame_base=divide_frame)
        
        # 延长分割帧停留
        self.add_pause_frames(divide_frame, count=self.divide_pause_frames)
        
        points_y_left = [p for p in points_y if p[0] <= mid_x]
        points_y_right = [p for p in points_y if p[0] > mid_x]
        
        indices_left = indices[:mid]
        indices_right = indices[mid:]
        
        dl, pair_l = self.closest_pair_recursive(
            points_x[:mid], points_y_left, indices_left, depth + 1
        )
        dr, pair_r = self.closest_pair_recursive(
            points_x[mid:], points_y_right, indices_right, depth + 1
        )
        
        if dl < dr:
            d = dl
            pair = pair_l
        else:
            d = dr
            pair = pair_r
        
        updated, old_pair = self.update_global_best(d, pair)
        
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
        
        if updated:
            self.add_text_transition('merge_update',
                                    text_content=f"合并阶段更新全局最佳: {d:.4f}",
                                    frame_base=merge_frame)
        
        # 修复：正确构建带区
        strip = []
        strip_indices = []
        for idx in indices:
            p = self.points[idx]
            if abs(p[0] - mid_x) < d:
                strip.append(p)
                strip_indices.append(idx)
        
        # 带区出现动画（传递strip_indices）
        self.add_smooth_transition('strip_appear', 
                                 mid_x=mid_x, delta=d, strip=strip, 
                                 strip_indices=strip_indices, depth=depth)
        
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
        # 延长带区帧停留
        self.add_pause_frames(strip_frame, count=self.strip_pause_frames)
        
        if len(strip) > 1:
            ds, pair_s = self.strip_closest(strip, d, strip_indices)
            
            if ds < d:
                d = ds
                pair = pair_s
                updated, old_pair = self.update_global_best(d, pair)
                
                strip_result_frame = {
                    'type': 'strip_result',
                    'mid_x': mid_x,
                    'delta': d,
                    'pair': pair,
                    'strip': strip,
                    'strip_indices': strip_indices,
                    'depth': depth,
                    'global_min_distance': self.global_min_distance,
                    'global_closest_pair': self.global_closest_pair
                }
                self.frames.append(strip_result_frame)
                
                self.add_text_transition('strip_discovery',
                                        text_content=f"带区发现更优解: {d:.4f}!",
                                        frame_base=strip_result_frame)
                
                # 延长带区结果帧停留
                self.add_pause_frames(strip_result_frame, count=max(1, self.pause_frames * 2))
        
        return d, pair
    
    def solve(self):
        """求解最近点对问题"""
        initial_frame = {
            'type': 'initial',
            'global_min_distance': self.global_min_distance,
            'global_closest_pair': self.global_closest_pair
        }
        self.frames.append(initial_frame)
        self.add_pause_frames(initial_frame, count=6)
        
        indices = list(range(self.n))
        points_x = sorted(zip(self.points, indices), key=lambda x: x[0][0])
        points_x, sorted_indices = zip(*points_x)
        points_x = np.array(points_x)
        sorted_indices = list(sorted_indices)
        
        points_y = sorted(self.points, key=lambda p: p[1])
        
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
        # 延长排序完成帧停留
        self.add_pause_frames(sorted_frame, count=max(1, self.pause_frames * 2))
        
        min_dist, pair = self.closest_pair_recursive(
            points_x, points_y, sorted_indices
        )
        
        completion_frame = {
            'type': 'completion',
            'global_min_distance': self.global_min_distance,
            'global_closest_pair': self.global_closest_pair
        }
        self.add_text_transition('algorithm_complete',
                                text_content=f"算法完成! 最短距离: {self.global_min_distance:.4f}",
                                frame_base=completion_frame)
        
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
        
        final_frame = {
            'type': 'final',
            'pair': self.global_closest_pair,
            'distance': self.global_min_distance,
            'global_min_distance': self.global_min_distance,
            'global_closest_pair': self.global_closest_pair
        }
        self.frames.append(final_frame)
        self.add_pause_frames(final_frame, count=10)
        
        return self.global_min_distance, self.global_closest_pair
    
    def render_beautiful_text(self, ax, text_content, text_alpha, text_type):
        """渲染美化的文字效果"""
        if not text_content:
            return
            
        if text_type == 'global_update':
            box_color = ModernColors.SUCCESS_BOX
            text_size = 14
            prefix = f"{Symbols.STAR} "
        elif text_type == 'strip_discovery':
            box_color = ModernColors.WARNING_BOX
            text_size = 13
            prefix = f"{Symbols.DIAMOND} "
        elif text_type == 'algorithm_start':
            box_color = ModernColors.INFO_BOX
            text_size = 15
            prefix = f"{Symbols.ARROW} "
        elif text_type == 'algorithm_complete':
            box_color = ModernColors.SUCCESS_BOX
            text_size = 16
            prefix = f"{Symbols.CHECK} "
        else:
            box_color = ModernColors.INFO_BOX
            text_size = 12
            prefix = f"{Symbols.INFO} "
        
        full_text = prefix + text_content
        
        ax.text(0.5, 0.85, full_text, transform=ax.transAxes,
               fontsize=text_size, fontweight='bold',
               horizontalalignment='center', verticalalignment='center',
               color=ModernColors.TEXT_PRIMARY, alpha=text_alpha,
               bbox=dict(boxstyle='round,pad=1.0', facecolor=box_color, 
                        alpha=text_alpha * 0.9, 
                        edgecolor=ModernColors.PRIMARY, linewidth=2))
    
    def create_animation(self, output_file='closest_pair_smooth.gif', fps=12, use_pillow=True):
        """创建超平滑动画"""
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor(ModernColors.BG_MAIN)
        
        def update(frame_idx):
            ax.clear()
            plt.rcParams['font.family'] = 'Microsoft YaHei'
            ax.set_facecolor(ModernColors.BG_CARD)
            
            frame = self.frames[frame_idx]
            frame_type = frame['type']
            
            global_min_dist = frame.get('global_min_distance', float('inf'))
            global_pair = frame.get('global_closest_pair', None)
            
            text_alpha = frame.get('text_alpha', 0)
            text_content = frame.get('text_content', '')
            text_type = frame.get('text_type', '')
            
            # 获取图例透明度
            legend_alpha = frame.get('legend_alpha', 1.0)
            
            # 绘制所有点（基础层）
            base_alpha = 0.7 if 'highlight' in frame_type or 'smooth' in frame_type else 0.8
            
            ax.scatter(self.points[:, 0], self.points[:, 1], 
                      c=ModernColors.POINT_NORMAL, s=60, alpha=base_alpha, 
                      edgecolors=ModernColors.PRIMARY, linewidth=0.8, zorder=1)
            
            # 绘制全局最佳点对（带淡入淡出）
            if global_pair and global_min_dist != float('inf'):
                p1, p2 = self.points[global_pair[0]], self.points[global_pair[1]]
                
                # 全局最佳的透明度随图例变化
                global_alpha = min(0.9, legend_alpha)
                
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                          c=ModernColors.POINT_BEST, s=200, alpha=0.3 * global_alpha, 
                          edgecolors='none', zorder=7)
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       color=ModernColors.LINE_BEST, linewidth=4, alpha=global_alpha, 
                       label=f"全局最佳 {global_min_dist:.4f}", zorder=8)
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                          c=ModernColors.POINT_BEST, s=140, 
                          edgecolors=ModernColors.PRIMARY, 
                          linewidth=2.5, zorder=9, alpha=0.95 * global_alpha)
            
            # 处理不同类型的帧
            title = ""
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
                
                line_alpha = 0.4 + 0.6 * alpha
                ax.axvline(x=mid_x, color=ModernColors.LINE_DIVIDE, linestyle='--', 
                          linewidth=3, alpha=line_alpha * legend_alpha, 
                          label='分割线')
                info_text = f"分割线 x = {mid_x:.3f}"
                
            elif frame_type in ['divide', 'divide_pause']:
                mid_x = frame.get('mid_x', 0.5)
                depth = frame.get('depth', 0)
                title += f"分治 - 分割 (深度 {depth})"
                ax.axvline(x=mid_x, color=ModernColors.LINE_DIVIDE, 
                          linestyle='--', linewidth=3, alpha=legend_alpha,
                          label='分割线')
                info_text = f"分割线 x = {mid_x:.3f}"
                
            elif frame_type == 'base_case_highlight':
                alpha = frame.get('alpha', 0)
                depth = frame.get('depth', 0)
                points = frame.get('points', [])
                title += f"基本情况处理中 (深度 {depth})"
                
                highlight_alpha = 0.3 + 0.7 * math.sin(alpha * math.pi)
                if len(points) > 0:
                    ax.scatter(points[:, 0], points[:, 1], 
                              c=ModernColors.POINT_HIGHLIGHT, s=80 + 40*highlight_alpha, 
                              edgecolors=ModernColors.PRIMARY, linewidth=1.5, 
                              zorder=5, alpha=highlight_alpha)
                
            elif frame_type in ['base_case', 'base_case_pause']:
                depth = frame.get('depth', 0)
                points = frame.get('points', [])
                pair = frame.get('pair')
                title += f"基本情况 (深度 {depth})"
                
                if len(points) > 0:
                    ax.scatter(points[:, 0], points[:, 1], 
                              c=ModernColors.POINT_HIGHLIGHT, s=100, 
                              edgecolors=ModernColors.PRIMARY, alpha=legend_alpha,
                              linewidth=1.5, zorder=5, label='处理中的点')
                
                if pair:
                    p1, p2 = self.points[pair[0]], self.points[pair[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=ModernColors.LINE_CURRENT, linewidth=2, alpha=legend_alpha,
                           label=f"局部最近点对 {frame['distance']:.4f}")
                    info_text = f"局部距离 = {frame['distance']:.4f}"
                
            elif frame_type in ['merge', 'merge_pause']:
                depth = frame.get('depth', 0)
                mid_x = frame.get('mid_x', 0.5)
                delta = frame.get('delta', 0)
                current_pair = frame.get('current_pair')
                title += f"合并阶段 (深度 {depth})"
                
                ax.axvline(x=mid_x, color=ModernColors.LINE_DIVIDE, 
                          linestyle='--', linewidth=3, label='分割线')
                
                if current_pair:
                    p1, p2 = self.points[current_pair[0]], self.points[current_pair[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=ModernColors.LINE_CURRENT, linewidth=2, 
                           label=f"当前最近点对 {delta:.4f}")
                
                info_text = f"当前δ = {delta:.4f}"
                
            elif frame_type == 'strip_appear_smooth':
                alpha = frame.get('alpha', 0)
                depth = frame.get('depth', 0)
                mid_x = frame.get('mid_x', 0.5)
                delta = frame.get('delta', 0)  # 当前宽度（渐变）
                target_delta = frame.get('target_delta', delta)  # 目标宽度
                strip = frame.get('strip', [])
                strip_indices = frame.get('strip_indices', [])
                title += f"带区出现中 (深度 {depth})"
                
                # 条带矩形淡入效果（宽度和透明度同时变化）
                rect_alpha = (0.15 + 0.15 * alpha) * legend_alpha
                rect = Rectangle((mid_x - delta, 0), 2 * delta, 1, 
                                alpha=rect_alpha, facecolor=ModernColors.STRIP_FILL, 
                                edgecolor=ModernColors.STRIP_BORDER, linewidth=2)
                ax.add_patch(rect)
                ax.axvline(x=mid_x, color=ModernColors.LINE_DIVIDE, 
                          linestyle='--', linewidth=3, alpha=legend_alpha,
                          label='分割线')
                
                if len(strip) > 0:
                    strip_array = np.array(strip)
                    point_alpha = alpha * legend_alpha
                    ax.scatter(strip_array[:, 0], strip_array[:, 1], 
                              c=ModernColors.POINT_HIGHLIGHT, s=60 + 40*alpha, 
                              edgecolors=ModernColors.STRIP_BORDER, linewidth=2, 
                              zorder=5, alpha=point_alpha, label='带区内点')
                
                info_text = f"目标δ = {target_delta:.4f}\n当前宽度 = {2*delta:.4f}"
                
            elif frame_type in ['strip', 'strip_pause']:
                depth = frame.get('depth', 0)
                mid_x = frame.get('mid_x', 0.5)
                delta = frame.get('delta', 0)
                strip = frame.get('strip', [])
                strip_indices = frame.get('strip_indices', [])
                title += f"检查带区 (深度 {depth})"
                
                rect = Rectangle((mid_x - delta, 0), 2 * delta, 1, 
                                alpha=0.3 * legend_alpha, facecolor=ModernColors.STRIP_FILL, 
                                edgecolor=ModernColors.STRIP_BORDER, linewidth=2)
                ax.add_patch(rect)
                ax.axvline(x=mid_x, color=ModernColors.LINE_DIVIDE, 
                          linestyle='--', linewidth=3, alpha=legend_alpha,
                          label='分割线')
                
                if len(strip) > 0:
                    strip_array = np.array(strip)
                    ax.scatter(strip_array[:, 0], strip_array[:, 1], 
                              c=ModernColors.POINT_HIGHLIGHT, s=100, 
                              edgecolors=ModernColors.STRIP_BORDER, 
                              linewidth=2, zorder=5, alpha=legend_alpha,
                              label='带区内点')
                
                info_text = f"δ = {delta:.4f}\n带区宽度 = {2*delta:.4f}\n带区点数 = {len(strip)}"
                
            elif frame_type in ['strip_result', 'strip_result_pause']:
                depth = frame.get('depth', 0)
                mid_x = frame.get('mid_x', 0.5)
                delta = frame.get('delta', 0)
                pair = frame.get('pair')
                strip = frame.get('strip', [])
                strip_indices = frame.get('strip_indices', [])
                title += f"带区发现更近点对! (深度 {depth})"
                
                rect = Rectangle((mid_x - delta, 0), 2 * delta, 1, 
                                alpha=0.3 * legend_alpha, facecolor=ModernColors.STRIP_FILL, 
                                edgecolor=ModernColors.STRIP_BORDER, linewidth=2)
                ax.add_patch(rect)
                ax.axvline(x=mid_x, color=ModernColors.LINE_DIVIDE, 
                          linestyle='--', linewidth=3, alpha=legend_alpha,
                          label='分割线')
                
                if len(strip) > 0:
                    strip_array = np.array(strip)
                    ax.scatter(strip_array[:, 0], strip_array[:, 1], 
                              c=ModernColors.POINT_HIGHLIGHT, s=100, 
                              edgecolors=ModernColors.STRIP_BORDER, 
                              linewidth=2, zorder=5, alpha=legend_alpha,
                              label='带区内点')
                
                if pair:
                    p1, p2 = self.points[pair[0]], self.points[pair[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=ModernColors.LINE_NEW, linewidth=3, alpha=legend_alpha,
                           label=f"新最近点对 {delta:.4f}", zorder=6)
                    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                              c=ModernColors.POINT_ACTIVE, s=150, 
                              edgecolors=ModernColors.PRIMARY, alpha=legend_alpha,
                              linewidth=2, zorder=7)
                
                info_text = f"新距离 = {delta:.4f}"
                
            elif frame_type == 'final_celebration':
                alpha = frame.get('alpha', 0)
                pulse = frame.get('pulse', 1)
                pair = frame.get('pair')
                distance = frame.get('distance', 0)
                title += "算法完成!"
                
                if pair:
                    p1, p2 = self.points[pair[0]], self.points[pair[1]]
                    
                    for i in range(3):
                        radius = (distance/2) * (1 + i * 0.5) * pulse
                        circle = Circle(p1, radius, fill=False, 
                                      edgecolor=ModernColors.LINE_NEW, linestyle=':', 
                                      linewidth=2-i*0.5, alpha=0.5-i*0.1)
                        ax.add_patch(circle)
                        circle = Circle(p2, radius, fill=False, 
                                      edgecolor=ModernColors.LINE_NEW, linestyle=':', 
                                      linewidth=2-i*0.5, alpha=0.5-i*0.1)
                        ax.add_patch(circle)
                    
                    line_width = 3 + 2 * pulse
                    point_size = 150 + 100 * pulse
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=ModernColors.LINE_NEW, linewidth=line_width, 
                           label=f"最近点对 {distance:.4f}", zorder=6)
                    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                              c=ModernColors.POINT_ACTIVE, s=point_size, 
                              edgecolors=ModernColors.PRIMARY, 
                              linewidth=3, zorder=7)
                
                info_text = f"最小距离 = {distance:.6f}"
                
            elif frame_type in ['final', 'final_pause']:
                title += "最终结果"
                pair = frame.get('pair')
                distance = frame.get('distance', 0)
                
                if pair:
                    p1, p2 = self.points[pair[0]], self.points[pair[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=ModernColors.LINE_NEW, linewidth=4, 
                           label=f"最近点对 {distance:.4f}", zorder=6)
                    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], 
                              c=ModernColors.POINT_ACTIVE, s=200, 
                              edgecolors=ModernColors.PRIMARY, 
                              linewidth=2, zorder=7)
                    
                    circle1 = Circle(p1, distance/2, fill=False, 
                                   edgecolor=ModernColors.LINE_NEW, linestyle=':', 
                                   linewidth=1.5, alpha=0.5)
                    circle2 = Circle(p2, distance/2, fill=False, 
                                   edgecolor=ModernColors.LINE_NEW, linestyle=':', 
                                   linewidth=1.5, alpha=0.5)
                    ax.add_patch(circle1)
                    ax.add_patch(circle2)
                
                    info_text = f"最小距离 = {distance:.6f}\n"
                    info_text += f"点对: ({p1[0]:.3f}, {p1[1]:.3f}) {Symbols.ARROW} ({p2[0]:.3f}, {p2[1]:.3f})"
            
            # 设置坐标轴
            ax.set_xlim(-0.08, 1.08)
            ax.set_ylim(-0.08, 1.08)
            ax.set_aspect('equal')
            
            ax.set_xlabel('X 坐标', fontsize=14, fontweight='500', 
                         color=ModernColors.TEXT_PRIMARY)
            ax.set_ylabel('Y 坐标', fontsize=14, fontweight='500', 
                         color=ModernColors.TEXT_PRIMARY)
            
            ax.set_title(title, fontsize=16, fontweight='600', 
                        color=ModernColors.TEXT_PRIMARY, pad=20)
            
            # 美化图例 - 修复：只在有标签时显示图例
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                # 去重图例
                by_label = dict(zip(labels, handles))
                legend = ax.legend(by_label.values(), by_label.keys(),
                                 loc='upper left', fontsize=11, 
                                 frameon=True, fancybox=True, shadow=True)
                legend.get_frame().set_facecolor(ModernColors.BG_CARD)
                legend.get_frame().set_edgecolor(ModernColors.SECONDARY)
                legend.get_frame().set_alpha(0.95)
            
            ax.grid(True, alpha=0.25, linewidth=0.8, color=ModernColors.SECONDARY)
            
            # 渲染文字动画
            if text_content and text_alpha > 0:
                self.render_beautiful_text(ax, text_content, text_alpha, text_type)
            
            # 美化信息文本框（也添加淡入淡出）
            if info_text:
                info_alpha = legend_alpha if 'smooth' in frame_type else 1.0
                ax.text(0.03, 0.03, info_text, transform=ax.transAxes,
                       fontsize=12, fontweight='500', verticalalignment='bottom',
                       color=ModernColors.TEXT_PRIMARY, alpha=info_alpha,
                       bbox=dict(boxstyle='round,pad=0.8', 
                                facecolor=ModernColors.INFO_BOX, 
                                alpha=0.9 * info_alpha, edgecolor=ModernColors.SECONDARY,
                                linewidth=1.5))
            
            # 美化的全局状态面板
            status_text = f"实时状态\n"
            if global_min_dist != float('inf'):
                status_text += f"最短距离: {global_min_dist:.6f}\n"
                if global_pair:
                    p1, p2 = self.points[global_pair[0]], self.points[global_pair[1]]
                    status_text += f"点对: P{global_pair[0]} {Symbols.ARROW} P{global_pair[1]}\n"
                    status_text += f"坐标: ({p1[0]:.3f}, {p1[1]:.3f})\n"
                    status_text += f"     ({p2[0]:.3f}, {p2[1]:.3f})"
            else:
                status_text += f"{Symbols.SEARCH} 搜索中..."
            
            status_color = ModernColors.SUCCESS_BOX if global_pair else ModernColors.INFO_BOX
            ax.text(0.97, 0.97, status_text, transform=ax.transAxes,
                   fontsize=12, fontweight='500', 
                   verticalalignment='top', horizontalalignment='right',
                   color=ModernColors.TEXT_PRIMARY,
                   bbox=dict(boxstyle='round,pad=1.0', 
                            facecolor=status_color, alpha=0.9,
                            edgecolor=ModernColors.PRIMARY, linewidth=2))
            
            # 添加水印
            ax.text(0.99, 0.01, f'{Symbols.STAR} 算法可视化--白晨均', transform=ax.transAxes,
                   fontsize=10, alpha=0.6, fontweight='300',
                   horizontalalignment='right', verticalalignment='bottom',
                   color=ModernColors.TEXT_SECONDARY)
        
        anim = animation.FuncAnimation(fig, update, frames=len(self.frames),
                                      interval=1000//fps, repeat=True)
        
        # 保存动画
        if use_pillow or output_file.endswith('.gif'):
            Writer = animation.writers['pillow']
            writer = Writer(fps=fps)
            anim.save(output_file, writer=writer, dpi=150)
        else:
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Beautiful Algorithm Visualization'), bitrate=2400)
                anim.save(output_file, writer=writer, dpi=150)
            except RuntimeError:
                Writer = animation.writers['pillow']
                writer = Writer(fps=fps)
                output_file = output_file.replace('.mp4', '.gif')
                anim.save(output_file, writer=writer, dpi=150)
        
        plt.close()
        print(f"超平滑动画已保存到: {output_file}")
        print(f"总帧数: {len(self.frames)}")
        print(f"帧率: {fps} FPS")
        print(f"预计时长: {len(self.frames)/fps:.1f} 秒")
 
 
def main():
    """主函数"""
    N = 45
    SEED = 42
    OUTPUT_FILE = './Assignment1/closest_pair_animation.gif'
    FPS = 15  # 提升到15 FPS，更流畅
    
    print(f"{Symbols.STAR} " + "=" * 58)
    print("   平面最近点对问题 - 分治算法可视化 (超平滑优化版)")
    print(f"{Symbols.STAR} " + "=" * 58)
    print(f"点数量: {N}")
    print(f"随机种子: {SEED}")
    print(f"帧率: {FPS} FPS")
    print()
    
    # 创建超平滑版可视化器
    visualizer = ClosestPairVisualizerSmooth(n=N, seed=SEED)
    min_dist, pair = visualizer.solve()
    
    print("算法求解完成！")
    print(f"最小距离: {min_dist:.6f}")
    if pair:
        p1, p2 = visualizer.points[pair[0]], visualizer.points[pair[1]]
        print(f"最近点对:")
        print(f"  点1: ({p1[0]:.6f}, {p1[1]:.6f})")
        print(f"  点2: ({p2[0]:.6f}, {p2[1]:.6f})")
    
    print(f"\n准备生成 {len(visualizer.frames)} 帧的动画...")
    print("正在生成动画...")
    visualizer.create_animation(output_file=OUTPUT_FILE, fps=FPS)
    print()
    print(f"{Symbols.CHECK} " + "=" * 58)
    print("完成！")
    print(f"{Symbols.CHECK} " + "=" * 58)
 
 
if __name__ == "__main__":
    main()