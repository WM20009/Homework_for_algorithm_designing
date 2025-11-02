"""
图像压缩问题 - 动态规划算法实现与可视化（完整版）
作者学号尾数: 8
算法: 使用动态规划求解最优分段方案，使总存储位数最小
包含自顶向下递归分析可视化（加分项）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import PillowWriter
from typing import List, Tuple
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ImageCompressionDP:
    """图像压缩动态规划算法"""
    
    def __init__(self, sequence: List[int]):
        """初始化"""
        self.sequence = sequence
        self.n = len(sequence)
        self.max_segment_length = 255
        
        # DP数组
        self.dp = [float('inf')] * (self.n + 1)
        self.dp[0] = 0
        
        # 记录最优分段方案
        self.parent = [-1] * (self.n + 1)
        self.segment_info = {}
        
        # 可视化记录
        self.visualization_steps = []
        
        # 递归分析记录
        self.recursive_memo = {}
        self.recursive_calls = []
        
    def calculate_segment_bits(self, start: int, end: int) -> int:
        """计算一个段的存储位数"""
        l = end - start
        if l == 0:
            return float('inf')
        
        segment = self.sequence[start:end]
        max_val = max(segment)
        min_val = min(segment)
        
        if max_val == min_val:
            b = 1
        else:
            b = math.ceil(math.log2(max_val - min_val + 1))
        
        bits = 8 + l + b * l
        return bits
    
    def solve(self):
        """动态规划求解"""
        for i in range(1, self.n + 1):
            step_candidates = []
            
            for j in range(max(0, i - self.max_segment_length), i):
                segment_bits = self.calculate_segment_bits(j, i)
                total_bits = self.dp[j] + segment_bits
                
                step_candidates.append({
                    'position': i,
                    'start': j,
                    'end': i,
                    'segment_bits': segment_bits,
                    'total_bits': total_bits,
                    'is_optimal': False,
                    'prev_dp': self.dp[j]
                })
                
                if total_bits < self.dp[i]:
                    self.dp[i] = total_bits
                    self.parent[i] = j
                    self.segment_info[i] = {
                        'start': j,
                        'end': i,
                        'bits': segment_bits
                    }
            
            for candidate in step_candidates:
                if candidate['start'] == self.parent[i] and candidate['end'] == i:
                    candidate['is_optimal'] = True
            
            self.visualization_steps.append({
                'type': 'step',
                'position': i,
                'candidates': step_candidates,
                'current_dp': self.dp[i],
                'optimal_from': self.parent[i]
            })
        
        return self.dp[self.n]
    
    def solve_recursive(self, target_position: int = None):
        """自顶向下递归求解（用于可视化）"""
        if target_position is None:
            target_position = self.n
        
        self.recursive_memo = {}
        self.recursive_calls = []
        
        result = self._solve_recursive_helper(target_position, call_depth=0, parent_call=None)
        return result
    
    def _solve_recursive_helper(self, i: int, call_depth: int, parent_call: int):
        """递归辅助函数"""
        call_id = len(self.recursive_calls)
        call_info = {
            'id': call_id,
            'position': i,
            'depth': call_depth,
            'parent': parent_call,
            'is_cached': i in self.recursive_memo,
            'result': None,
            'children': []
        }
        
        if i == 0:
            call_info['result'] = 0
            call_info['is_base_case'] = True
            self.recursive_calls.append(call_info)
            return 0
        
        if i in self.recursive_memo:
            call_info['result'] = self.recursive_memo[i]
            call_info['from_cache'] = True
            self.recursive_calls.append(call_info)
            return self.recursive_memo[i]
        
        call_info['is_base_case'] = False
        call_info['from_cache'] = False
        self.recursive_calls.append(call_info)
        
        min_cost = float('inf')
        best_j = -1
        
        for j in range(max(0, i - self.max_segment_length), i):
            cost_j = self._solve_recursive_helper(j, call_depth + 1, call_id)
            segment_cost = self.calculate_segment_bits(j, i)
            total_cost = cost_j + segment_cost
            
            self.recursive_calls[call_id]['children'].append({
                'start': j,
                'end': i,
                'segment_cost': segment_cost,
                'total_cost': total_cost
            })
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_j = j
        
        self.recursive_memo[i] = min_cost
        self.recursive_calls[call_id]['result'] = min_cost
        self.recursive_calls[call_id]['best_from'] = best_j
        
        return min_cost
    
    def get_optimal_segments(self) -> List[Tuple[int, int]]:
        """回溯获取最优分段方案"""
        segments = []
        current = self.n
        
        while current > 0:
            start = self.parent[current]
            segments.append((start, current))
            current = start
        
        segments.reverse()
        return segments


class ImageCompressionVisualizer:
    """图像压缩可视化器"""
    
    def __init__(self, solver: ImageCompressionDP, output_file: str = "image_compression.gif"):
        self.solver = solver
        self.output_file = output_file
        
        # 动画参数
        self.expand_frames = 15
        self.compare_frames = 12
        self.update_frames = 10
        self.backtrack_frames = 15
        self.pause_frames = 8
        self.phase_transition = 6
        self.smooth_frames = 12
        
        # 配色方案
        self.colors = {
            'background': '#F8F9FA',
            'sequence_bar': '#E9ECEF',
            'expand_color': '#42A5F5',
            'compare_color': '#FFA726',
            'update_color': '#66BB6A',
            'backtrack_color': '#AB47BC',
            'optimal': '#4CAF50',
            'text': '#212529',
            'grid': '#DEE2E6',
            'phase_bg': '#FFFFFF'
        }
        
    def create_animation(self):
        """创建完整动画"""
        fig = plt.figure(figsize=(20, 12), facecolor=self.colors['background'])
        writer = PillowWriter(fps=15)
        
        with writer.saving(fig, self.output_file, dpi=100):
            self._animate_intro(fig, writer)
            self._animate_recursive_analysis(fig, writer)
            self._animate_dp_process_with_phases(fig, writer)
            self._animate_backtrack(fig, writer)
            self._animate_summary(fig, writer)
        
        plt.close()
        print(f"\n动画已生成: {self.output_file}")
    
    def _animate_intro(self, fig, writer):
        """动画开场"""
        for frame in range(self.pause_frames * 2):
            fig.clear()
            alpha = min(1.0, frame / 6)
            
            ax = fig.add_subplot(111)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            ax.text(5, 7.5, '图像压缩问题 - 动态规划算法', 
                   ha='center', va='center', fontsize=32, 
                   fontweight='bold', color=self.colors['text'], alpha=alpha)
            
            ax.text(5, 6.5, f'序列长度: {self.solver.n} | 最大段长: {self.solver.max_segment_length}',
                   ha='center', va='center', fontsize=18,
                   color=self.colors['text'], alpha=alpha * 0.8)
            
            if frame >= 6:
                phase_alpha = min(1.0, (frame - 6) / 6)
                phases = [
                    ('1. 扩展候选段', self.colors['expand_color']),
                    ('2. 比较计算代价', self.colors['compare_color']),
                    ('3. 更新最优解', self.colors['update_color']),
                    ('4. 回溯构造方案', self.colors['backtrack_color'])
                ]
                
                for idx, (text, color) in enumerate(phases):
                    y_pos = 4.5 - idx * 0.8
                    ax.text(5, y_pos, text, ha='center', va='center',
                           fontsize=16, color=color, fontweight='bold',
                           alpha=phase_alpha,
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor='white', edgecolor=color,
                                   linewidth=2, alpha=phase_alpha))
            
            writer.grab_frame()
    
    def _animate_recursive_analysis(self, fig, writer):
        """递归分析可视化（加分项）"""
        # 介绍页面
        for frame in range(self.pause_frames * 2):
            fig.clear()
            alpha = min(1.0, frame / 8)
            
            ax = fig.add_subplot(111)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            ax.text(5, 7, '自顶向下递归分析', ha='center', va='center',
                   fontsize=32, fontweight='bold', 
                   color='#9C27B0', alpha=alpha)
            
            ax.text(5, 5.5, '记忆化递归求解过程', ha='center', va='center',
                   fontsize=20, color=self.colors['text'], alpha=alpha)
            
            if frame >= 8:
                desc_alpha = min(1.0, (frame - 8) / 6)
                desc_text = "递归思路: solve(i) = min{solve(j) + cost(j,i)}\n\n从目标位置开始递归分解\n缓存已计算的子问题\n避免重复计算"
                ax.text(5, 3, desc_text, ha='center', va='center',
                       fontsize=14, color=self.colors['text'], alpha=desc_alpha,
                       bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                                edgecolor='#9C27B0', linewidth=2))
            
            writer.grab_frame()
        
        # 使用小规模示例
        demo_length = min(30, self.solver.n)
        print(f"\n  执行递归分析 (示例长度: {demo_length})...")
        self.solver.solve_recursive(demo_length)
        
        key_calls = self._select_key_recursive_calls(demo_length)
        self._animate_recursion_tree(fig, writer, key_calls, demo_length)
        self._animate_memoization(fig, writer, demo_length)
    
    def _select_key_recursive_calls(self, target: int):
        """选择关键的递归调用"""
        all_calls = self.solver.recursive_calls
        key_calls = []
        depth_samples = {}
        
        for call in all_calls:
            depth = call['depth']
            
            if call['position'] == target or call.get('is_base_case', False):
                key_calls.append(call)
                continue
            
            if depth not in depth_samples:
                depth_samples[depth] = []
            depth_samples[depth].append(call)
        
        for depth, calls in sorted(depth_samples.items()):
            samples = calls[:min(3 if depth <= 3 else 2, len(calls))]
            key_calls.extend(samples)
        
        key_calls.sort(key=lambda x: x['id'])
        return key_calls[:25]
    
    def _animate_recursion_tree(self, fig, writer, key_calls, target):
        """展示递归调用树"""
        for call_idx, call in enumerate(key_calls):
            for frame in range(self.smooth_frames + self.pause_frames // 2):
                fig.clear()
                progress = min(1.0, frame / self.smooth_frames)
                
                gs = fig.add_gridspec(2, 2, height_ratios=[0.6, 0.4],
                                     width_ratios=[0.6, 0.4],
                                     hspace=0.25, wspace=0.20)
                
                ax_tree = fig.add_subplot(gs[0, :])
                self._draw_recursion_tree(ax_tree, key_calls, call_idx, progress, target)
                
                ax_info = fig.add_subplot(gs[1, 0])
                self._draw_current_call_info(ax_info, call, progress)
                
                ax_cache = fig.add_subplot(gs[1, 1])
                self._draw_cache_state(ax_cache, call_idx, key_calls)
                
                fig.text(0.5, 0.97, f'递归调用树 - 调用 {call_idx + 1}/{len(key_calls)}',
                        ha='center', va='top', fontsize=18, fontweight='bold',
                        color='#9C27B0')
                
                writer.grab_frame()
    
    def _draw_recursion_tree(self, ax, key_calls, current_idx, progress, target):
        """绘制递归树"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        num_show = int((current_idx + 1) * progress)
        positions = {}
        depth_counts = {}
        
        for i, call in enumerate(key_calls[:num_show]):
            depth = call['depth']
            if depth not in depth_counts:
                depth_counts[depth] = 0
            
            x = 1 + depth_counts[depth] * 0.7
            y = 9 - depth * 1.7
            
            positions[call['id']] = (x, y)
            depth_counts[depth] += 1
        
        for i, call in enumerate(key_calls[:num_show]):
            if call['id'] not in positions:
                continue
            
            x, y = positions[call['id']]
            
            if i == current_idx:
                color = '#FF5722'
                alpha = 0.9
                size = 3400
            elif call.get('from_cache', False):
                color = '#2196F3'
                alpha = 0.7
                size = 3000
            elif call.get('is_base_case', False):
                color = '#4CAF50'
                alpha = 0.7
                size = 3000
            else:
                color = '#9C27B0'
                alpha = 0.6
                size = 2700
            
            ax.scatter([x], [y], s=size, c=color, alpha=alpha, 
                      edgecolors='white', linewidths=2, zorder=10)
            
            label = f"solve({call['position']})"
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=12, color='white', fontweight='bold', zorder=11)
            
            if call['parent'] is not None:
                parent_pos = positions.get(call['parent'])
                if parent_pos:
                    ax.plot([parent_pos[0], x], [parent_pos[1], y],
                           color='#BDBDBD', linewidth=2, alpha=0.5, zorder=1)
        
        legend_y = 1
        legend_items = [
            ('当前', '#FF5722'),
            ('缓存', '#2196F3'),
            ('基础', '#4CAF50'),
            ('计算', '#9C27B0')
        ]
        
        for idx, (label, color) in enumerate(legend_items):
            x_pos = 7.5 + (idx % 2) * 1.5
            y_pos = legend_y - (idx // 2) * 0.4
            ax.scatter([x_pos], [y_pos], s=200, c=color, alpha=0.7,
                      edgecolors='white', linewidths=1)
            ax.text(x_pos + 0.3, y_pos, label, ha='left', va='center',
                   fontsize=9, color=self.colors['text'])
    
    def _draw_current_call_info(self, ax, call, progress):
        """绘制当前调用信息"""
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        alpha = progress
        
        ax.text(5, 9, '当前递归调用', ha='center', va='top',
               fontsize=14, fontweight='bold', color='#FF5722', alpha=alpha)
        
        y_pos = 7.5
        line_height = 1.2
        
        info_items = [
            f"位置: solve({call['position']})",
            f"深度: {call['depth']}",
        ]
        
        if call.get('is_base_case'):
            info_items.append("类型: 基础情况")
            info_items.append(f"返回: 0")
        elif call.get('from_cache'):
            info_items.append("类型: 缓存命中")
            info_items.append(f"返回: {call['result']}")
        else:
            info_items.append("类型: 首次计算")
            if call['result'] is not None:
                info_items.append(f"结果: {call['result']}")
        
        for item in info_items:
            ax.text(5, y_pos, item, ha='center', va='center',
                   fontsize=12, color=self.colors['text'], 
                   fontweight='bold', alpha=alpha)
            y_pos -= line_height
        
        bg_rect = patches.FancyBboxPatch((0.5, 1), 9, 8.5,
                                        boxstyle="round,pad=0.3",
                                        facecolor='white',
                                        edgecolor='#FF5722',
                                        linewidth=2, alpha=alpha * 0.8)
        ax.add_patch(bg_rect)
    
    def _draw_cache_state(self, ax, current_idx, key_calls):
        """绘制缓存状态"""
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        ax.text(5, 9, '记忆化缓存', ha='center', va='top',
               fontsize=14, fontweight='bold', color='#2196F3')
        
        cached_positions = set()
        for i, call in enumerate(key_calls[:current_idx + 1]):
            if call.get('from_cache') or (call['result'] is not None and not call.get('is_base_case')):
                cached_positions.add(call['position'])
        
        ax.text(5, 7.5, f'已缓存位置数: {len(cached_positions)}',
               ha='center', va='center', fontsize=12,
               color=self.colors['text'], fontweight='bold')
        
        y_pos = 6
        for pos in sorted(cached_positions)[:8]:
            result = self.solver.recursive_memo.get(pos, '?')
            if result != '?':
                ax.text(5, y_pos, f"memo[{pos}] = {result}",
                       ha='center', va='center', fontsize=10,
                       color=self.colors['text'])
                y_pos -= 0.8
        
        if len(cached_positions) > 8:
            ax.text(5, y_pos, f'... 还有 {len(cached_positions) - 8} 个',
                   ha='center', va='center', fontsize=9,
                   color=self.colors['text'], style='italic')
        
        bg_rect = patches.FancyBboxPatch((0.5, 1), 9, 8.5,
                                        boxstyle="round,pad=0.3",
                                        facecolor='white',
                                        edgecolor='#2196F3',
                                        linewidth=2, alpha=0.8)
        ax.add_patch(bg_rect)
    
    def _animate_memoization(self, fig, writer, target):
        """展示记忆化效果对比"""
        for frame in range(self.pause_frames * 2):
            fig.clear()
            
            ax = fig.add_subplot(111)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            alpha = min(1.0, frame / 8)
            
            ax.text(5, 8.5, '记忆化效果分析', ha='center', va='center',
                   fontsize=26, fontweight='bold', color='#2196F3', alpha=alpha)
            
            total_calls = len(self.solver.recursive_calls)
            cached_calls = sum(1 for c in self.solver.recursive_calls if c.get('from_cache', False))
            unique_positions = len(self.solver.recursive_memo)
            
            y_pos = 6.5
            stats = [
                f"目标位置: solve({target})",
                "",
                f"总递归调用次数: {total_calls}",
                f"缓存命中次数: {cached_calls}",
                f"实际计算次数: {total_calls - cached_calls}",
                "",
                f"不同子问题数: {unique_positions}",
                f"重复避免率: {(cached_calls/total_calls*100):.1f}%" if total_calls > 0 else "0%"
            ]
            
            for stat in stats:
                if stat:
                    ax.text(5, y_pos, stat, ha='center', va='center',
                           fontsize=15, color=self.colors['text'],
                           fontweight='bold', alpha=alpha)
                y_pos -= 0.7
            
            conclusion = "记忆化避免了大量重复计算！"
            ax.text(5, 2, conclusion, ha='center', va='center',
                   fontsize=16, color='#4CAF50', fontweight='bold',
                   alpha=alpha,
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                            edgecolor='#4CAF50', linewidth=3))
            
            writer.grab_frame()
    
    def _animate_dp_process_with_phases(self, fig, writer):
        """展示DP求解过程"""
        step_interval = max(1, len(self.solver.visualization_steps) // 25)
        key_steps = [i for i in range(0, len(self.solver.visualization_steps), step_interval)]
        key_steps.append(len(self.solver.visualization_steps) - 1)
        
        self.key_steps = key_steps
        self.total_key_steps = len(key_steps)
        
        for step_idx in key_steps:
            step = self.solver.visualization_steps[step_idx]
            self._animate_phase_expand(fig, writer, step, step_idx)
            self._animate_phase_compare(fig, writer, step, step_idx)
            self._animate_phase_update(fig, writer, step, step_idx)
    
    def _animate_phase_expand(self, fig, writer, step, step_idx):
        """阶段1: 扩展候选段"""
        position = step['position']
        candidates = step['candidates']
        
        for frame in range(self.expand_frames):
            fig.clear()
            progress = frame / (self.expand_frames - 1) if self.expand_frames > 1 else 1.0
            
            gs = fig.add_gridspec(3, 2, height_ratios=[0.12, 0.5, 0.35], 
                                 width_ratios=[0.7, 0.3], hspace=0.25, wspace=0.15)
            
            ax_phase = fig.add_subplot(gs[0, :])
            self._draw_phase_indicator(ax_phase, 'expand', progress)
            
            ax_main = fig.add_subplot(gs[1, :])
            self._draw_sequence_expanded(ax_main, position, candidates, progress, 'expand')
            
            ax_table = fig.add_subplot(gs[2, 0])
            self._draw_candidates_table(ax_table, candidates, progress, 'expand')
            
            ax_dp = fig.add_subplot(gs[2, 1])
            self._draw_dp_mini(ax_dp, position)
            
            fig.text(0.5, 0.96, f' 位置 {position}',
                    ha='center', va='top', fontsize=20, fontweight='bold',
                    color=self.colors['text'])
            
            writer.grab_frame()
        
        for _ in range(self.pause_frames // 2):
            writer.grab_frame()
    
    def _animate_phase_compare(self, fig, writer, step, step_idx):
        """阶段2: 比较计算代价"""
        position = step['position']
        candidates = step['candidates']
        
        for frame in range(self.compare_frames):
            fig.clear()
            progress = frame / (self.compare_frames - 1) if self.compare_frames > 1 else 1.0
            
            gs = fig.add_gridspec(3, 2, height_ratios=[0.12, 0.5, 0.35],
                                 width_ratios=[0.7, 0.3], hspace=0.25, wspace=0.15)
            
            ax_phase = fig.add_subplot(gs[0, :])
            self._draw_phase_indicator(ax_phase, 'compare', progress)
            
            ax_main = fig.add_subplot(gs[1, :])
            self._draw_sequence_expanded(ax_main, position, candidates, 1.0, 'compare', progress)
            
            ax_table = fig.add_subplot(gs[2, 0])
            self._draw_candidates_table(ax_table, candidates, progress, 'compare')
            
            ax_dp = fig.add_subplot(gs[2, 1])
            self._draw_dp_mini(ax_dp, position)
            
            fig.text(0.5, 0.96, f' 位置 {position}',
                    ha='center', va='top', fontsize=20, fontweight='bold',
                    color=self.colors['text'])
            
            writer.grab_frame()
        
        for _ in range(self.pause_frames // 2):
            writer.grab_frame()
    
    def _animate_phase_update(self, fig, writer, step, step_idx):
        """阶段3: 更新最优解"""
        position = step['position']
        candidates = step['candidates']
        
        for frame in range(self.update_frames + self.pause_frames):
            fig.clear()
            
            if frame < self.update_frames:
                highlight_alpha = 0.4 + 0.3 * abs(np.sin(frame * np.pi / 5))
            else:
                highlight_alpha = 0.7
            
            gs = fig.add_gridspec(3, 2, height_ratios=[0.12, 0.5, 0.35],
                                 width_ratios=[0.7, 0.3], hspace=0.25, wspace=0.15)
            
            ax_phase = fig.add_subplot(gs[0, :])
            self._draw_phase_indicator(ax_phase, 'update', 1.0)
            
            ax_main = fig.add_subplot(gs[1, :])
            self._draw_sequence_optimal_only(ax_main, position, candidates, highlight_alpha)
            
            ax_table = fig.add_subplot(gs[2, 0])
            self._draw_candidates_table(ax_table, candidates, 1.0, 'update', highlight_alpha)
            
            ax_dp = fig.add_subplot(gs[2, 1])
            self._draw_dp_mini(ax_dp, position, highlight_alpha)
            
            fig.text(0.5, 0.96, f' 位置 {position}',
                    ha='center', va='top', fontsize=20, fontweight='bold',
                    color=self.colors['text'])
            
            writer.grab_frame()
    
    def _draw_phase_indicator(self, ax, current_phase, progress):
        """绘制阶段指示器"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        phases = [
            ('扩展', 'expand', self.colors['expand_color']),
            ('比较', 'compare', self.colors['compare_color']),
            ('更新', 'update', self.colors['update_color']),
        ]
        
        for idx, (name, phase_key, color) in enumerate(phases):
            x_start = 1 + idx * 3
            
            if phase_key == current_phase:
                alpha = 0.3 + 0.4 * progress
                linewidth = 3
            else:
                alpha = 0.15
                linewidth = 1
            
            rect = patches.FancyBboxPatch((x_start - 0.4, 0.15), 2.5, 0.7,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor=color,
                                         alpha=alpha, linewidth=linewidth)
            ax.add_patch(rect)
            
            text_alpha = 1.0 if phase_key == current_phase else 0.5
            ax.text(x_start + 1.05, 0.5, f'{idx + 1}. {name}',
                   ha='center', va='center', fontsize=14,
                   color=color if phase_key == current_phase else self.colors['text'],
                   fontweight='bold' if phase_key == current_phase else 'normal',
                   alpha=text_alpha)
    
    def _draw_sequence_expanded(self, ax, position, candidates, progress, phase, compare_progress=0):
        """扩大显示序列和分段"""
        show_range = 60
        start_pos = max(0, position - show_range)
        end_pos = min(self.solver.n, position + 15)
        
        step = max(1, (end_pos - start_pos) // 50)
        x_coords = list(range(start_pos, end_pos, step))
        y_coords = [self.solver.sequence[i] for i in x_coords]
        
        ax.bar(x_coords, y_coords, width=step*0.7,
              color=self.colors['sequence_bar'], alpha=0.4,
              edgecolor=self.colors['text'], linewidth=0.3)
        
        if phase == 'expand':
            num_show = int(len(candidates) * progress)
            total_candidates = len(candidates)
            
            for i, candidate in enumerate(candidates[:num_show]):
                alpha = 0.15
                show_label = (i < 2 or i >= total_candidates - 2 or 
                             i % max(1, total_candidates // 5) == 0)
                self._draw_segment_box(ax, candidate, self.colors['expand_color'], 
                                      alpha, step, False, show_label)
        
        elif phase == 'compare':
            compare_idx = int(len(candidates) * compare_progress)
            for i, candidate in enumerate(candidates):
                if i == compare_idx:
                    self._draw_segment_box(ax, candidate, self.colors['compare_color'], 
                                          0.3, step, False, True)
                else:
                    self._draw_segment_box(ax, candidate, self.colors['expand_color'], 
                                          0.1, step, False, False)
        
        ax.axvline(x=position, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
        ax.text(position, 270, f'位置 {position}', ha='center', va='bottom',
               fontsize=11, color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='red', linewidth=2))
        
        ax.set_xlim(start_pos - 5, end_pos + 5)
        ax.set_ylim(0, 285)
        ax.set_xlabel('序列位置', fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('灰度值', fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.grid(True, alpha=0.25, color=self.colors['grid'], axis='y')
        ax.set_facecolor(self.colors['background'])
    
    def _draw_segment_box(self, ax, candidate, color, alpha, step=1, is_optimal=False, show_label=True):
        """绘制段的框"""
        start = candidate['start']
        end = candidate['end']
        
        if is_optimal:
            rect = patches.Rectangle((start - step*0.4, 0), end - start + step*0.4, 260,
                                     linewidth=4, edgecolor=color,
                                     facecolor=color, alpha=alpha * 0.3)
        else:
            rect = patches.Rectangle((start - step*0.4, 0), end - start + step*0.4, 260,
                                     linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        if show_label:
            mid_x = (start + end) / 2
            bits = candidate['segment_bits']
            ax.text(mid_x, 245, f"{bits}b", ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                            edgecolor='white', linewidth=1.5, alpha=0.9))
    
    def _draw_sequence_optimal_only(self, ax, position, candidates, highlight_alpha):
        """只显示最优段"""
        show_range = 60
        start_pos = max(0, position - show_range)
        end_pos = min(self.solver.n, position + 15)
        
        step = max(1, (end_pos - start_pos) // 50)
        x_coords = list(range(start_pos, end_pos, step))
        y_coords = [self.solver.sequence[i] for i in x_coords]
        
        ax.bar(x_coords, y_coords, width=step*0.7,
              color=self.colors['sequence_bar'], alpha=0.4,
              edgecolor=self.colors['text'], linewidth=0.3)
        
        for candidate in candidates:
            if candidate['is_optimal']:
                start = candidate['start']
                end = candidate['end']
                
                rect = patches.Rectangle((start - step*0.4, 0), end - start + step*0.4, 260,
                                         linewidth=5, edgecolor=self.colors['update_color'],
                                         facecolor=self.colors['update_color'], alpha=0.2)
                ax.add_patch(rect)
                
                mid_x = (start + end) / 2
                bits = candidate['segment_bits']
                ax.text(mid_x, 245, f"{bits}bits", ha='center', va='center',
                       fontsize=10, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor=self.colors['update_color'],
                                edgecolor='white', linewidth=2, alpha=0.95))
        
        ax.axvline(x=position, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
        ax.text(position, 270, f'位置 {position}', ha='center', va='bottom',
               fontsize=11, color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='red', linewidth=2))
        
        ax.set_xlim(start_pos - 5, end_pos + 5)
        ax.set_ylim(0, 285)
        ax.set_xlabel('序列位置', fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('灰度值', fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.grid(True, alpha=0.25, color=self.colors['grid'], axis='y')
        ax.set_facecolor(self.colors['background'])
    
    def _draw_candidates_table(self, ax, candidates, progress, phase, highlight_alpha=0.5):
        """绘制候选段信息表"""
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        phase_names = {'expand': '候选段列表', 'compare': '代价比较', 'update': '最优选择'}
        title = phase_names.get(phase, '候选段')
        ax.text(5, 9.2, title, ha='center', va='top', fontsize=14,
               fontweight='bold', color=self.colors['text'])
        
        max_show = 5
        if phase == 'expand':
            show_candidates = candidates[:int(len(candidates) * progress)][:max_show]
        else:
            optimal_idx = next((i for i, c in enumerate(candidates) if c['is_optimal']), 0)
            start_idx = max(0, optimal_idx - 2)
            show_candidates = candidates[start_idx:start_idx + max_show]
        
        y_start = 8.0
        for idx, candidate in enumerate(show_candidates):
            y_pos = y_start - idx * 1.5
            
            if candidate['is_optimal']:
                bg_color = self.colors['update_color']
                bg_alpha = highlight_alpha if phase == 'update' else 0.3
            elif phase == 'compare':
                bg_color = self.colors['compare_color']
                bg_alpha = 0.2
            else:
                bg_color = self.colors['expand_color']
                bg_alpha = 0.15
            
            rect = patches.FancyBboxPatch((0.3, y_pos - 0.5), 9.4, 1.0,
                                         boxstyle="round,pad=0.1",
                                         facecolor=bg_color, alpha=bg_alpha,
                                         edgecolor=bg_color, linewidth=2)
            ax.add_patch(rect)
            
            seg_len = candidate['end'] - candidate['start']
            text = f"[{candidate['start']:3d}, {candidate['end']:3d})  L={seg_len:3d}  " \
                   f"Bits={candidate['segment_bits']:4d}  Total={int(candidate['total_bits']):5d}"
            
            text_color = 'white' if candidate['is_optimal'] and phase == 'update' else self.colors['text']
            ax.text(5, y_pos, text, ha='center', va='center',
                   fontsize=9, color=text_color,
                   fontweight='bold' if candidate['is_optimal'] else 'normal')
        
        if len(candidates) > max_show:
            ax.text(5, y_start - max_show * 1.5 - 0.3,
                   f'... 还有 {len(candidates) - len(show_candidates)} 个候选段',
                   ha='center', va='top', fontsize=9,
                   color=self.colors['text'], style='italic')
    
    def _draw_dp_mini(self, ax, position, highlight_alpha=0.5):
        """绘制DP状态"""
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        ax.text(5, 9.2, 'DP状态', ha='center', va='top',
               fontsize=14, fontweight='bold', color=self.colors['text'])
        
        show_positions = [max(0, position - i) for i in range(min(6, position + 1))]
        show_positions.reverse()
        
        y_start = 7.5
        for idx, pos in enumerate(show_positions):
            y_pos = y_start - idx * 1.2
            
            dp_val = self.solver.dp[pos]
            dp_str = "∞" if dp_val == float('inf') else f"{int(dp_val)}"
            
            is_current = (pos == position)
            if is_current:
                bg_color = self.colors['update_color']
                bg_alpha = highlight_alpha
                text_color = 'white'
                text_weight = 'bold'
            else:
                bg_color = self.colors['sequence_bar']
                bg_alpha = 0.3
                text_color = self.colors['text']
                text_weight = 'normal'
            
            rect = patches.FancyBboxPatch((1.5, y_pos - 0.4), 7, 0.8,
                                         boxstyle="round,pad=0.05",
                                         facecolor=bg_color, alpha=bg_alpha,
                                         edgecolor=bg_color, linewidth=2 if is_current else 1)
            ax.add_patch(rect)
            
            text = f"DP[{pos:3d}] = {dp_str:>6s}"
            ax.text(5, y_pos, text, ha='center', va='center',
                   fontsize=10, color=text_color,
                   fontweight=text_weight)
    
    def _animate_backtrack(self, fig, writer):
        """回溯展示最优解"""
        optimal_segments = self.solver.get_optimal_segments()
        
        for frame in range(self.phase_transition):
            fig.clear()
            alpha = frame / self.phase_transition
            
            ax = fig.add_subplot(111)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            ax.text(5, 5, '阶段4: 回溯构造最优方案', ha='center', va='center',
                   fontsize=32, fontweight='bold', 
                   color=self.colors['backtrack_color'], alpha=alpha)
            
            writer.grab_frame()
        
        for seg_idx, (start, end) in enumerate(optimal_segments):
            for frame in range(self.backtrack_frames + self.pause_frames):
                fig.clear()
                
                progress = min(1.0, frame / self.backtrack_frames)
                
                gs = fig.add_gridspec(2, 1, height_ratios=[0.58, 0.32], 
                                     hspace=0.35, top=0.92, bottom=0.06)
                
                ax_main = fig.add_subplot(gs[0])
                self._draw_backtrack_sequence(ax_main, optimal_segments, seg_idx, progress)
                
                ax_info = fig.add_subplot(gs[1])
                self._draw_segment_info_panel(ax_info, optimal_segments, seg_idx, progress)
                
                title = f'阶段4: 回溯 - 第 {seg_idx + 1}/{len(optimal_segments)} 段'
                fig.text(0.5, 0.97, title, ha='center', va='top',
                        fontsize=18, fontweight='bold', color=self.colors['backtrack_color'])
                
                writer.grab_frame()
    
    def _draw_backtrack_sequence(self, ax, segments, current_idx, progress):
        """绘制回溯序列"""
        step = max(1, self.solver.n // 100)
        x_coords = list(range(0, self.solver.n, step))
        y_coords = [self.solver.sequence[i] for i in x_coords]
        
        ax.bar(x_coords, y_coords, width=step*0.7,
              color=self.colors['sequence_bar'], alpha=0.35,
              edgecolor=self.colors['text'], linewidth=0.2)
        
        if current_idx < len(segments):
            start, end = segments[current_idx]
            alpha = 0.5 + 0.3 * progress
            self._draw_final_segment(ax, start, end, self.colors['backtrack_color'], 
                                    alpha, current_idx + 1, step)
        
        ax.set_xlim(-5, self.solver.n + 5)
        ax.set_ylim(0, 285)
        ax.set_xlabel('序列位置', fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('灰度值', fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.grid(True, alpha=0.25, color=self.colors['grid'], axis='y')
        ax.set_facecolor(self.colors['background'])
        ax.tick_params(axis='x', labelsize=11, pad=8)
        ax.tick_params(axis='y', labelsize=11)
    
    def _draw_final_segment(self, ax, start, end, color, alpha, seg_num, step=1):
        """绘制最终分段"""
        rect = patches.Rectangle((start - step*0.5, 0), end - start + step*0.5, 270,
                                 linewidth=4, edgecolor=color,
                                 facecolor=color, alpha=alpha * 0.25)
        ax.add_patch(rect)
        
        mid_x = (start + end) / 2
        segment_len = end - start
        
        segment_data = self.solver.sequence[start:end]
        min_val = min(segment_data)
        max_val = max(segment_data)
        b = 1 if max_val == min_val else math.ceil(math.log2(max_val - min_val + 1))
        bits = 8 + segment_len + b * segment_len
        
        label_text = f'段{seg_num}\n[{start},{end})\nL={segment_len}\n{bits}bits'
        
        ax.text(mid_x, 235, label_text, ha='center', va='center',
               fontsize=10, color='white', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color,
                        edgecolor='white', linewidth=2.5, alpha=0.95))
    
    def _draw_segment_info_panel(self, ax, segments, current_idx, progress):
        """绘制分段信息面板"""
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        if current_idx >= len(segments):
            return
        
        current_segment = segments[current_idx]
        start, end = current_segment
        segment_len = end - start
        
        segment_data = self.solver.sequence[start:end]
        min_val = min(segment_data)
        max_val = max(segment_data)
        
        b = 1 if max_val == min_val else math.ceil(math.log2(max_val - min_val + 1))
        bits = 8 + segment_len + b * segment_len
        
        alpha = min(1.0, progress + 0.3)
        
        y_pos = 7.5
        line_height = 1.2
        
        ax.text(5, y_pos, f'当前段编号: 第 {current_idx + 1} 段', 
               ha='center', va='center', fontsize=15, 
               color=self.colors['backtrack_color'], fontweight='bold', alpha=alpha)
        
        y_pos -= line_height * 1.5
        ax.plot([1, 9], [y_pos, y_pos], color=self.colors['backtrack_color'], 
               linewidth=2, alpha=alpha * 0.5)
        
        y_pos -= line_height
        ax.text(5, y_pos, f'段范围: [{start}, {end})     段长度: {segment_len}',
               ha='center', va='center', fontsize=14,
               color=self.colors['text'], fontweight='bold', alpha=alpha)
        
        y_pos -= line_height
        ax.text(5, y_pos, f'灰度范围: [{min_val}, {max_val}]     编码位数: {b} bit/像素',
               ha='center', va='center', fontsize=14,
               color=self.colors['text'], fontweight='bold', alpha=alpha)
        
        y_pos -= line_height
        ax.text(5, y_pos, f'该段存储: {bits} bits',
               ha='center', va='center', fontsize=14,
               color=self.colors['text'], fontweight='bold', alpha=alpha)
        
        y_pos -= line_height * 1.5
        ax.plot([1, 9], [y_pos, y_pos], color=self.colors['backtrack_color'], 
               linewidth=2, alpha=alpha * 0.5)
        
        y_pos -= line_height
        ax.text(5, y_pos, f'已回溯: {current_idx + 1}/{len(segments)} 段',
               ha='center', va='center', fontsize=14,
               color=self.colors['backtrack_color'], fontweight='bold', alpha=alpha)
        
        bg_rect = patches.FancyBboxPatch((0.5, 1), 9, 8.5,
                                        boxstyle="round,pad=0.3",
                                        facecolor='white',
                                        edgecolor=self.colors['backtrack_color'],
                                        linewidth=3, alpha=alpha * 0.9)
        ax.add_patch(bg_rect)
    
    def _animate_summary(self, fig, writer):
        """结果总结"""
        optimal_segments = self.solver.get_optimal_segments()
        total_bits = int(self.solver.dp[self.solver.n])
        
        for frame in range(self.pause_frames * 3):
            fig.clear()
            alpha = min(1.0, frame / 8)
            
            ax_main = fig.add_axes([0.08, 0.42, 0.84, 0.48])
            self._draw_final_result(ax_main, optimal_segments, alpha)
            
            ax_stats = fig.add_axes([0.08, 0.05, 0.84, 0.32])
            self._draw_statistics(ax_stats, optimal_segments, total_bits, alpha)
            
            fig.text(0.5, 0.96, '算法完成 - 最优压缩方案', ha='center', va='top',
                    fontsize=22, fontweight='bold', color=self.colors['optimal'], alpha=alpha)
            
            writer.grab_frame()
    
    def _draw_final_result(self, ax, segments, alpha):
        """绘制最终结果"""
        step = max(1, self.solver.n // 100)
        x_coords = list(range(0, self.solver.n, step))
        y_coords = [self.solver.sequence[i] for i in x_coords]
        
        ax.bar(x_coords, y_coords, width=step*0.7,
              color=self.colors['sequence_bar'], alpha=alpha * 0.4,
              edgecolor=self.colors['text'], linewidth=0.2)
        
        colors_palette = [
            self.colors['optimal'], 
            self.colors['expand_color'],
            self.colors['compare_color'],
            self.colors['backtrack_color']
        ]
        
        for idx, (start, end) in enumerate(segments):
            color = colors_palette[idx % len(colors_palette)]
            self._draw_final_segment(ax, start, end, color, alpha * 0.75, idx + 1, step)
        
        ax.set_xlim(-5, self.solver.n + 5)
        ax.set_ylim(0, 285)
        ax.set_xlabel('序列位置', fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('灰度值', fontsize=14, color=self.colors['text'], fontweight='bold')
        ax.set_title('完整最优分段方案', fontsize=16, color=self.colors['text'], 
                    fontweight='bold', pad=15)
        ax.grid(True, alpha=0.25, color=self.colors['grid'], axis='y')
        ax.set_facecolor(self.colors['background'])
    
    def _draw_statistics(self, ax, segments, total_bits, alpha):
        """绘制统计信息"""
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        num_segments = len(segments)
        avg_segment_len = sum(end - start for start, end in segments) / num_segments
        
        segment_lengths = [end - start for start, end in segments]
        min_seg_len = min(segment_lengths)
        max_seg_len = max(segment_lengths)
        
        original_bits = self.solver.n * 8
        compression_ratio = (1 - total_bits / original_bits) * 100
        
        stats_text = f"""
        ═══════════════════════════════════════════════════════════
        
        序列长度: {self.solver.n:<6d}              分段数量: {num_segments:<4d}
        
        原始存储: {original_bits:<8d} bits      压缩后: {total_bits:<8d} bits
        
        压缩率: {compression_ratio:>6.2f}%
        
        平均段长: {avg_segment_len:>6.1f}            段长范围: [{min_seg_len}, {max_seg_len}]
        
        ═══════════════════════════════════════════════════════════
        """
        
        ax.text(5, 5, stats_text, ha='center', va='center',
               fontsize=14, color=self.colors['text'], alpha=alpha,
               bbox=dict(boxstyle='round,pad=1.0', facecolor='white',
                        edgecolor=self.colors['optimal'], linewidth=4, alpha=alpha * 0.95),
               fontweight='bold')


def generate_test_sequence(n: int, seed: int = 42) -> List[int]:
    """生成测试用灰度序列"""
    np.random.seed(seed)
    
    sequence = []
    position = 0
    
    while position < n:
        wave_type = np.random.choice(['flat', 'linear', 'sine', 'random'])
        segment_length = np.random.randint(10, 30)
        
        if position + segment_length > n:
            segment_length = n - position
        
        base_value = np.random.randint(50, 200)
        
        if wave_type == 'flat':
            values = [base_value + np.random.randint(-5, 5) for _ in range(segment_length)]
        elif wave_type == 'linear':
            slope = np.random.choice([-2, -1, 1, 2])
            values = [np.clip(base_value + slope * i, 0, 255) for i in range(segment_length)]
        elif wave_type == 'sine':
            amplitude = np.random.randint(20, 50)
            frequency = np.random.uniform(0.1, 0.3)
            values = [np.clip(int(base_value + amplitude * np.sin(frequency * i)), 0, 255) 
                     for i in range(segment_length)]
        else:
            values = [np.clip(base_value + np.random.randint(-30, 30), 0, 255) 
                     for _ in range(segment_length)]
        
        sequence.extend(values)
        position += segment_length
    
    return sequence[:n]


def main():
    """主函数"""
    print("=" * 60)
    print("图像压缩问题 - 动态规划算法（完整版）")
    print("=" * 60)
    
    SEQUENCE_LENGTH = 150
    RANDOM_SEED = 42
    OUTPUT_FILE = "image_compression.gif"
    
    print(f"\n配置参数:")
    print(f"  序列长度: {SEQUENCE_LENGTH}")
    print(f"  随机种子: {RANDOM_SEED}")
    print(f"  输出文件: {OUTPUT_FILE}")
    
    print("\n[1/4] 生成测试序列...")
    sequence = generate_test_sequence(SEQUENCE_LENGTH, RANDOM_SEED)
    print(f"  序列生成完成，长度: {len(sequence)}")
    print(f"  灰度值范围: [{min(sequence)}, {max(sequence)}]")
    
    print("\n[2/4] 执行动态规划算法...")
    solver = ImageCompressionDP(sequence)
    min_bits = solver.solve()
    optimal_segments = solver.get_optimal_segments()
    
    print(f"  算法执行完成")
    print(f"  最小存储位数: {min_bits}")
    print(f"  最优分段数量: {len(optimal_segments)}")
    
    print("\n  最优分段方案:")
    for i, (start, end) in enumerate(optimal_segments[:10]):
        seg_len = end - start
        seg_data = sequence[start:end]
        seg_range = f"[{min(seg_data)}, {max(seg_data)}]"
        print(f"    段 {i+1}: [{start:3d}, {end:3d}) | 长度: {seg_len:3d} | 灰度范围: {seg_range}")
    
    if len(optimal_segments) > 10:
        print(f"    ... 还有 {len(optimal_segments) - 10} 段")
    
    original_bits = SEQUENCE_LENGTH * 8
    compression_ratio = (1 - min_bits / original_bits) * 100
    print(f"\n  压缩效果:")
    print(f"    原始存储: {original_bits} bits")
    print(f"    压缩后: {min_bits} bits")
    print(f"    压缩率: {compression_ratio:.2f}%")
    
    print("\n[3/4] 执行递归分析（可视化）...")
    solver.solve_recursive()
    print("  递归分析完成")
    
    print("\n[4/4] 生成可视化动画...")
    visualizer = ImageCompressionVisualizer(solver, OUTPUT_FILE)
    visualizer.create_animation()
    print("  动画生成完成")
    
    print("\n任务完成！")

if __name__=='__main__':
    main()