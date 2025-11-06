#week4作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"
target_len = len(sentence)
# 获取所有词典
word_list = list(Dict.keys())
# 最终全切分的结果
target = []

def full_segment(index, current_segmentation):
    """
    递归进行全切分 (回溯法)。

    Args:
        index (int): 当前开始切分的起始位置（在 sentence 中的下标）。
        current_segmentation (list): 到目前为止已经切分好的词语列表。
    """
    
    # 成功条件：如果当前起始位置达到了句子的末尾，说明找到一个完整的切分路径
    if index == len(sentence):
        target.append(list(current_segmentation))
        return

    # 遍历所有可能的词长，从 1 到 最大的合法词长（或句子的剩余长度）
    # 在这个例子中，最长词是 '见分歧' 或 '有意见'/'经常'/'分歧' (长度为 3)
    max_word_len = 3 
    
    for length in range(1, max_word_len + 1):
        end_index = index + length
        
        # 确保不会越界
        if end_index > len(sentence):
            continue

        # 提取候选词
        candidate_word = sentence[index:end_index]
        
        # 检查候选词是否在词典中
        if candidate_word in word_list:
            # 将候选词添加到当前切分结果中
            current_segmentation.append(candidate_word)
            
            # 递归: 从下一个位置开始继续切分
            full_segment(end_index, current_segmentation)
            
            # 回溯: 撤销选择，以便尝试下一个候选词
            current_segmentation.pop()



#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # 获取词典的最大长度，做为切分的最大长度
    max_length = len(max(word_list, key=len))
    # 获取所有可能切分的词组
    cut_words = []
    for word in sentence:
        chars = []
        i = 1
        while i <= max_length:
            work_index = sentence.find(word)
            if(sentence[work_index:work_index+i] in word_list):
                # print(sentence[work_index:work_index+i])
                chars.append(sentence[work_index:work_index+i])
            i=i+1
        cut_words.append(chars)
    
    print("获取所有可能切分的词组：", cut_words)
    # 启动搜索
    print("开始搜索所有组合路径...")
    full_segment(0, [])
    print("搜索完成，最终结果：", target)
    return target

if __name__ == "__main__":
    all_cut(sentence, Dict)

#目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]


