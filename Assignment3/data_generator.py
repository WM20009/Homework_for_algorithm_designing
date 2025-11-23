"""
增强版数据生成脚本
专为展示贪心算法局部最优缺陷而设计
生成测试数据特点：
- n ≥ 20个活动（默认25个）
- 包含"贪心陷阱"：一个长时中权活动阻挡10个短时高权活动
- 确保DP最优解 ≥ 10个活动，贪心最优率 ≤ 95%
- 活动持续时间和权值多样性
- 混合分布：密集、稀疏区域结合
"""

import random
import json
import numpy as np
from weighted_scheduling_code import Activity, WeightedIntervalScheduler

def generate_enhanced_data(n=25, seed=42, save_to_file=True):
    """
    增强版数据生成 - 循环验证确保质量
    核心设计：创造一个贪心算法会失败的明确场景
    
    参数:
        n: 活动数量
        seed: 随机种子
        save_to_file: 是否保存到文件
    
    返回:
        activities: 活动对象列表（已按结束时间排序）
    """
    iteration = 0
    
    while True:
        current_seed = seed + iteration
        random.seed(current_seed)
        np.random.seed(current_seed)
        
        activities = []
        
        # ===== 阶段1：设置贪心陷阱 =====
        # 活动0：持续时间长、权重中等（60）
        # 贪心算法会优先选择它（结束时间较早）
        # 但它会阻挡后面10个高权重短活动（总权重239）
        activities.append(Activity(0, 5, 35, 60))  # 陷阱活动
        
        # ===== 阶段2：与陷阱冲突的更优组合 =====
        # 这3个活动总权重93，虽然单个不如陷阱，但为DP提供选择空间
        activities.append(Activity(1, 10, 25, 35))
        activities.append(Activity(2, 20, 30, 30))
        activities.append(Activity(3, 25, 35, 28))
        
        # ===== 阶段3：黄金区间（10个高权重短活动）=====
        # 这些活动在陷阱结束后才开始，总权重239
        # 如果贪心选了陷阱（0-35），就无法选择这些活动
        # 这是展示贪心算法缺陷的核心设计
        gold_activities = [
            (4, 35, 42, 22),   # 持续7，权重22
            (5, 42, 48, 25),   # 持续6，权重25
            (6, 48, 55, 23),   # 持续7，权重23
            (7, 55, 61, 26),   # 持续6，权重26
            (8, 61, 68, 24),   # 持续7，权重24
            (9, 68, 74, 27),   # 持续6，权重27
            (10, 74, 81, 25),  # 持续7，权重25
            (11, 81, 87, 28),  # 持续6，权重28
            (12, 87, 93, 26),  # 持续6，权重26
            (13, 93, 100, 29), # 持续7，权重29
        ]
        
        for id, start, end, weight in gold_activities:
            activities.append(Activity(id, start, end, weight))
        
        # ===== 阶段4：补充随机活动 =====
        # 增加数据多样性，避免过于规则
        # 部分与陷阱或黄金区间重叠，部分独立
        for i in range(14, n):
            if i < n * 0.6:
                # 前半段：集中在陷阱区域
                start = random.randint(0, 45)
                duration = random.randint(8, 15)
                weight = random.randint(18, 32)
            else:
                # 后半段：分布在黄金区间附近
                start = random.randint(35, 90)
                duration = random.randint(10, 20)
                weight = random.randint(20, 40)
            
            end = start + duration
            activities.append(Activity(i, start, end, weight))
        
        # 按结束时间排序（算法要求）
        activities_sorted = sorted(activities, key=lambda a: a.end)
        
        # 重新分配ID以匹配算法内部索引
        for i, act in enumerate(activities_sorted):
            act.id = i
        
        # ===== 阶段5：质量验证 =====
        # 确保生成的数据满足作业要求
        scheduler = WeightedIntervalScheduler(activities_sorted)
        dp_selected, dp_weight, _ = scheduler.dp_solve()
        greedy_selected, greedy_weight, _ = scheduler.greedy_solve()
        
        diff = dp_weight - greedy_weight
        ratio = greedy_weight / dp_weight
        
        print(f"验证数据质量: 种子={current_seed}")
        print(f"  DP: {dp_weight} (活动数={len(dp_selected)})")
        print(f"  贪心: {greedy_weight} (活动数={len(greedy_selected)})")
        print(f"  差值={diff}, 比例={ratio:.2%}")
        
        # 检查是否满足所有条件
        if len(dp_selected) >= 10 and diff >= 15 and ratio <= 0.95:
            print("✓ 数据质量达标！")
            print(f"  - DP选择活动数: {len(dp_selected)} (要求≥10)")
            print(f"  - 权重差值: {diff} (要求≥15)")
            print(f"  - 贪心最优率: {ratio:.1%} (要求≤95%)")
            
            # 保存元数据
            if save_to_file:
                with open('test_data.json', 'w', encoding='utf-8') as f:
                    json.dump({
                        'seed': current_seed,
                        'original_seed': seed,
                        'n': n,
                        'activities': [
                            {
                                'id': a.id,
                                'start': a.start,
                                'end': a.end,
                                'weight': a.weight,
                                'duration': a.duration
                            }
                            for a in activities_sorted
                        ],
                        'greedy_result': {
                            'selected_ids': [activities_sorted[i].id for i in greedy_selected],
                            'weight': greedy_weight,
                            'activity_count': len(greedy_selected)
                        },
                        'dp_result': {
                            'selected_ids': [activities_sorted[i].id for i in dp_selected],
                            'weight': dp_weight,
                            'activity_count': len(dp_selected)
                        },
                        'quality_metrics': {
                            'weight_difference': diff,
                            'greedy_optimality_ratio': ratio,
                            'dp_activity_count': len(dp_selected)
                        }
                    }, f, indent=2, ensure_ascii=False)
                print(f"✓ 数据已保存到 test_data.json")
            
            return activities_sorted
        
        else:
            print("✗ 数据质量不达标，重新生成...")
            iteration += 1
            if iteration > 50:
                print("警告：超过50次尝试，返回最佳结果")
                return activities_sorted

def load_activities(filename='test_data.json'):
    """从文件加载活动数据并转换为Activity对象"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    activities = []
    for act_data in data['activities']:
        activities.append(Activity(
            act_data['id'],
            act_data['start'],
            act_data['end'],
            act_data['weight']
        ))
    
    # 确保按结束时间排序
    activities_sorted = sorted(activities, key=lambda a: a.end)
    return activities_sorted

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 20 + "增强版数据生成器")
    print("=" * 70)
    print("\n设计目标：创造贪心算法明确的失败场景")
    print("策略：长时中权活动(60)阻挡10个短时高权活动(总239)")
    print("要求：DP≥10个活动，差值≥15，贪心最优率≤95%\n")
    
    activities = generate_enhanced_data(n=25, seed=42, save_to_file=True)
    
    print("\n" + "=" * 70)
    print("数据生成完成！")
    print("=" * 70)