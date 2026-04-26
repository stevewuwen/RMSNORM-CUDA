import re
import os

def count_chinese_characters(file_path):
    """
    统计文件中的中文字符数量
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在。")
        return

    try:
        # 以 UTF-8 编码读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 使用正则表达式匹配中文字符
            # \u4e00-\u9fff 是最常用的中文字符 Unicode 范围
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
            chinese_list = chinese_pattern.findall(content)
            
            count = len(chinese_list)
            print(f"文件路径: {file_path}")
            print(f"统计结果: 共有 {count} 个中文字符。")
            return count

    except UnicodeDecodeError:
        print("错误：文件解码失败。请确保文件是 UTF-8 编码。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # 获取用户输入的文件路径
    path = "thesis_draft.md"
    
    # 如果路径带引号（拖入文件时常见），去掉引号
    path = path.replace('"', '').replace("'", "")
    
    count_chinese_characters(path)