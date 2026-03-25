"""
GP树生成模块
包含用于生成GP表达式树的函数
"""

import random
import sys
import warnings
from inspect import isclass

# 定义任何类型的类型名称
__type__ = object


def genFull(pset, min_, max_, type_=None):
    """生成一个表达式，其中每个叶子节点在*min*和*max*之间具有相同的深度"""
    def condition(height, depth):
        """当深度等于高度时，表达式生成停止"""
        return depth == height
    return generate(pset, min_, max_, condition, type_)


def genGrow(pset, min_, max_, type_=None):
    """生成一个表达式，其中每个叶子节点在*min*和*max*之间可能具有不同的深度"""
    def condition(height, depth):
        """当深度等于高度或随机确定节点应为终端时，表达式生成停止"""
        return depth == height or depth >= min_
    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """使用原始集合*pset*生成表达式
    一半时间使用genGrow生成表达式，另一半时间使用genFull生成表达式"""
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def generate(pset, min_, max_, condition, type_=__type__):
    """生成一个树作为列表的列表。树从根到叶子构建，当满足条件时停止生长。"""
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        # 在树的底部
        if condition(height, depth):
            # 尝试找到一个终端
            try:
                term = random.choice(pset.terminals[type_])
                if isclass(term):
                    term = term()
                expr.append(term)
            except:
                # 没有终端适合，所以将深度拉回一层，开始寻找原语
                try:
                    depth -= 1
                    prim = random.choice(pset.primitives[type_])
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth + 1, arg))
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("gp.generate函数尝试添加类型'%s'的原语，但没有可用的。" % (type_,), traceback)
        # 不在树的底部
        else:
            # 检查原语
            try:
                prim = random.choice(pset.primitives[type_])
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
            except:
                # 没有原语适合，所以检查终端
                try:
                    term = random.choice(pset.terminals[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("gp.generate函数尝试添加类型'%s'的终端，但没有可用的。" % (type_,), traceback)
                if isclass(term):
                    term = term()
                expr.append(term)
    return expr


def genHalfAndHalfMD(pset, min_, max_, type_=None):
    """基于genHalfAndHalf但限制树节点数量≤80"""
    expr = genHalfAndHalf(pset, min_, max_, type_=type_)
    while len(expr) > 80:
        expr = genHalfAndHalf(pset, min_, max_, type_=type_)
    return expr


def genFullMD(pset, min_, max_, type_=None):
    """基于genFull但限制树节点数量≤80"""
    expr = genFull(pset, min_, max_, type_=type_)
    while len(expr) > 80:
        expr = genFull(pset, min_, max_, type_=type_)
    return expr


# 创建一个命名空间对象，用于兼容旧的导入方式
class GPRestrict:
    """GP树生成限制类，提供各种树生成方法"""
    genFull = staticmethod(genFull)
    genGrow = staticmethod(genGrow)
    genHalfAndHalf = staticmethod(genHalfAndHalf)
    genHalfAndHalfMD = staticmethod(genHalfAndHalfMD)
    genFullMD = staticmethod(genFullMD)


# 创建全局实例，用于兼容旧的导入方式：from ... import gp_restrict
gp_restrict = GPRestrict()

