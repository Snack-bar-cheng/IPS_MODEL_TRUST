"""
Ridge公式生成模块
"""


def generate_ridge_formula(model, gp_expression, target_name='y'):
    """
    生成Ridge回归的最终公式
    
    输入：
        model: 训练好的RidgeCV模型
        gp_expression: GP表达式字符串
        target_name: 目标变量名称，默认为'y'
    
    输出：
        str: Ridge回归公式
    """
    try:
        # 获取Ridge回归的系数和截距
        coefs = model.coef_
        intercept = float(model.intercept_)
        
        # 解析GP表达式，提取High_N中的各个子特征
        # 例如：High_10(f1, f2, ..., f10) -> [f1, f2, ..., f10]
        if 'High_' in gp_expression:
            # 找到High_N(...)的内容
            start_idx = gp_expression.find('(')
            end_idx = gp_expression.rfind(')')
            if start_idx != -1 and end_idx != -1:
                # 提取括号内的内容
                inner_content = gp_expression[start_idx+1:end_idx]
                
                # 简单解析：通过括号匹配分割特征
                features = []
                depth = 0
                current_feature = ""
                
                for char in inner_content:
                    if char == '(':
                        depth += 1
                        current_feature += char
                    elif char == ')':
                        depth -= 1
                        current_feature += char
                    elif char == ',' and depth == 0:
                        features.append(current_feature.strip())
                        current_feature = ""
                    else:
                        current_feature += char
                
                # 添加最后一个特征
                if current_feature.strip():
                    features.append(current_feature.strip())
                
                # 生成公式
                if len(features) == len(coefs):
                    formula_parts = []
                    for i, (coef, feature) in enumerate(zip(coefs, features)):
                        coef_val = float(coef)
                        if i == 0:
                            formula_parts.append(f"{coef_val:.6f} * ({feature})")
                        else:
                            sign = " + " if coef_val >= 0 else " "
                            formula_parts.append(f"{sign}{coef_val:.6f} * ({feature})")
                    
                    # 处理截距的符号
                    intercept_str = f" + {intercept:.6f}" if intercept >= 0 else f" {intercept:.6f}"
                    formula = f"{target_name} = " + "".join(formula_parts) + intercept_str
                    return formula
        
        # 如果解析失败，返回简化版本
        formula_parts = []
        for i, coef in enumerate(coefs):
            coef_val = float(coef)
            if i == 0:
                formula_parts.append(f"{coef_val:.6f} * feature_{i+1}")
            else:
                sign = " + " if coef_val >= 0 else " "
                formula_parts.append(f"{sign}{coef_val:.6f} * feature_{i+1}")
        
        # 处理截距的符号
        intercept_str = f" + {intercept:.6f}" if intercept >= 0 else f" {intercept:.6f}"
        formula = f"{target_name} = " + "".join(formula_parts) + intercept_str
        return formula
        
    except Exception as e:
        return f"无法生成Ridge公式: {str(e)}"
