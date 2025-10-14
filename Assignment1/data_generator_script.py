"""
数据生成脚本 - 生成满足作业要求的测试数据
Data Generator for Closest Pair Problem
"""

import numpy as np
import json
import random


def generate_points(n=50, seed=42, mid_line=0.5):
    """
    生成满足作业要求的点集
    
    要求：
    1. 点数 n >= 36
    2. 点位于分界线两侧，左右两侧点数相差为 10%-20%
    3. 在中线±10%宽度内至少有 30% 的点
    
    :param n: 点的数量
    :param seed: 随机种子
    :param mid_line: 分界线位置（默认0.5）
    :return: 点集数组
    """
    random.seed(seed)
    np.random.seed(seed)
    
    band_width = 0.1  # ±10%宽度
    points_in_band = int(n * 0.35)  # 35%的点在带区（满足至少30%的要求）
    remaining_points = n - points_in_band
    
    # 左侧点数约为剩余点的55-65%，保证左右相差10%-20%
    left_ratio = random.uniform(0.55, 0.65)
    points_left_side = int(remaining_points * left_ratio)
    points_right_side = remaining_points - points_left_side
    
    points = []
    
    # 生成带区内的点
    print(f"生成带区内点: {points_in_band}个")
    for _ in range(points_in_band):
        x = np.random.uniform(mid_line - band_width, mid_line + band_width)
        y = np.random.uniform(0, 1)
        points.append([x, y])
    
    # 生成左侧的点
    print(f"生成左侧点: {points_left_side}个")
    for _ in range(points_left_side):
        x = np.random.uniform(0, mid_line - band_width)
        y = np.random.uniform(0, 1)
        points.append([x, y])
    
    # 生成右侧的点
    print(f"生成右侧点: {points_right_side}个")
    for _ in range(points_right_side):
        x = np.random.uniform(mid_line + band_width, 1)
        y = np.random.uniform(0, 1)
        points.append([x, y])
    
    # 验证生成的数据
    points_array = np.array(points)
    left_count = np.sum(points_array[:, 0] < mid_line)
    right_count = np.sum(points_array[:, 0] > mid_line)
    band_count = np.sum(np.abs(points_array[:, 0] - mid_line) < band_width)
    
    print(f"\n数据验证:")
    print(f"  总点数: {len(points)}")
    print(f"  左侧点数: {left_count} ({left_count/len(points)*100:.1f}%)")
    print(f"  右侧点数: {right_count} ({right_count/len(points)*100:.1f}%)")
    print(f"  带区点数: {band_count} ({band_count/len(points)*100:.1f}%)")
    print(f"  左右差异: {abs(left_count - right_count)/len(points)*100:.1f}%")
    
    return np.array(points)


def save_points_to_file(points, filename, seed):
    """保存点集到文件"""
    data = {
        'seed': seed,
        'n': len(points),
        'points': points.tolist()
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n数据已保存到: {filename}")


def load_points_from_file(filename):
    """从文件加载点集"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return np.array(data['points']), data['seed']


def verify_data_requirements(points, mid_line=0.5):
    """验证数据是否满足作业要求"""
    n = len(points)
    band_width = 0.1
    
    left_count = np.sum(points[:, 0] < mid_line)
    right_count = np.sum(points[:, 0] > mid_line)
    band_count = np.sum(np.abs(points[:, 0] - mid_line) < band_width)
    
    # 要求1: n >= 36
    req1 = n >= 36
    
    # 要求2: 左右点数相差 10%-20%
    diff_ratio = abs(left_count - right_count) / n
    req2 = 0.10 <= diff_ratio <= 0.20
    
    # 要求3: 带区内至少30%的点
    band_ratio = band_count / n
    req3 = band_ratio >= 0.30
    
    print("\n=== 数据要求验证 ===")
    print(f"要求1 - 点数 >= 36: {'✓' if req1 else '✗'} (n={n})")
    print(f"要求2 - 左右相差 10%-20%: {'✓' if req2 else '✗'} (相差{diff_ratio*100:.1f}%)")
    print(f"要求3 - 带区至少30%: {'✓' if req3 else '✗'} (带区{band_ratio*100:.1f}%)")
    
    all_passed = req1 and req2 and req3
    print(f"\n总体: {'全部通过 ✓' if all_passed else '部分未通过 ✗'}")
    
    return all_passed


def generate_multiple_datasets():
    """生成多组测试数据集"""
    test_cases = [
        {'n': 36, 'seed': 42, 'name': 'small'},
        {'n': 50, 'seed': 123, 'name': 'medium'},
        {'n': 100, 'seed': 456, 'name': 'large'},
        {'n': 200, 'seed': 789, 'name': 'xlarge'},
    ]
    
    print("=" * 70)
    print("生成多组测试数据集")
    print("=" * 70)
    
    for case in test_cases:
        print(f"\n{'='*70}")
        print(f"测试集: {case['name']} (n={case['n']}, seed={case['seed']})")
        print(f"{'='*70}")
        
        points = generate_points(n=case['n'], seed=case['seed'])
        filename = f"data_{case['name']}_n{case['n']}_seed{case['seed']}.json"
        save_points_to_file(points, filename, case['seed'])
        verify_data_requirements(points)


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--multiple':
            # 生成多组数据集
            generate_multiple_datasets()
        elif sys.argv[1] == '--verify':
            # 验证指定文件
            if len(sys.argv) < 3:
                print("用法: python data_generator.py --verify <filename>")
                return
            filename = sys.argv[2]
            points, seed = load_points_from_file(filename)
            print(f"从文件加载: {filename}")
            print(f"随机种子: {seed}")
            verify_data_requirements(points)
        else:
            print("未知选项，使用 --multiple 或 --verify")
    else:
        # 默认：生成单组数据
        print("=" * 70)
        print("数据生成脚本 - 平面最近点对问题")
        print("=" * 70)
        
        # 默认参数
        N = 50
        SEED = 42
        
        print(f"\n生成参数:")
        print(f"  点数 n = {N}")
        print(f"  随机种子 = {SEED}")
        print(f"  分界线 x = 0.5")
        print(f"  带区宽度 = ±0.1")
        print()
        
        # 生成点集
        points = generate_points(n=N, seed=SEED)
        
        # 保存到文件
        filename = f"data_n{N}_seed{SEED}.json"
        save_points_to_file(points, filename, SEED)
        
        # 验证数据
        verify_data_requirements(points)
        
        print("\n" + "=" * 70)
        print("完成！")
        print("=" * 70)
        print("\n提示:")
        print("  - 使用 --multiple 生成多组测试数据")
        print("  - 使用 --verify <filename> 验证数据文件")


if __name__ == "__main__":
    main()