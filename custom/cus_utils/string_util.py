import re

######## 字符串相关工具类 ##########

def find_first_digit_position(text):
    """查找字符串中第一个数字的位置 """
    
    match = re.search(r"\d", text)
    if match:
        return match.start()
    else:
        return None