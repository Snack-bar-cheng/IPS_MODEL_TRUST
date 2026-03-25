import math

"""
单元测试：自适应扩展决策逻辑（精英滑动窗口）
逻辑摘自 executor/gp_evolution.py 中的自适应策略核心判断：
 - 使用窗口首尾计算平均增长率 avg_growth
 - 使用最近两代的差值 last_diff
 - 条件：avg_growth > growth_threshold 且 last_diff > 0 时继续探索；否则扩展
"""


def should_expand(window, growth_threshold, window_size):
    """
    基于精英平均fitness窗口的决策：
    - window: list[float]，长度>=window_size
    - 返回 (expand: bool, avg_growth: float, last_diff: float)
    """
    if len(window) < window_size:
        # 窗口不足时默认扩展
        return True, math.nan, math.nan

    first_val = window[-window_size]
    last_val = window[-1]
    prev_val = window[-2] if len(window) >= 2 else last_val

    avg_growth = (last_val - first_val) / (window_size - 1) if window_size > 1 else 0.0
    last_diff = last_val - prev_val

    expand = not (avg_growth > growth_threshold and last_diff > 0)
    return expand, avg_growth, last_diff


def test_continue_explore_when_growth_positive():
    # 连续增长，平均增长>阈值，最近一代仍在上升 -> 不扩展
    window = [10.0, 12.0, 14.5]  # avg_growth=(14.5-10)/2=2.25 > 1.0, last_diff=2.5>0
    expand, avg_growth, last_diff = should_expand(window, growth_threshold=1.0, window_size=3)
    assert expand is False
    assert avg_growth > 1.0
    assert last_diff > 0


def test_expand_when_growth_slows_or_negative():
    # 增长不足或下降 -> 扩展
    window = [20.0, 20.5, 20.6]  # avg_growth=0.3, last_diff=0.1 -> 低增长，触发扩展
    expand, _, _ = should_expand(window, growth_threshold=1.0, window_size=3)
    assert expand is True

    window = [20.0, 19.5, 19.0]  # 下降 -> 扩展
    expand, _, _ = should_expand(window, growth_threshold=1.0, window_size=3)
    assert expand is True


def test_expand_when_window_insufficient():
    # 窗口不足时默认扩展
    window = [15.0, 16.0]  # 长度2，window_size=3
    expand, _, _ = should_expand(window, growth_threshold=1.0, window_size=3)
    assert expand is True


if __name__ == "__main__":
    # 直接运行文件以便手工快速验证
    test_continue_explore_when_growth_positive()
    test_expand_when_growth_slows_or_negative()
    test_expand_when_window_insufficient()
    print("dynamic policy tests passed")

