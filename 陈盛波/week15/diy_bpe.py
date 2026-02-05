# 扩展词表
expand_vocab_1 = {} # key:合并后的token，value:合并前的token
expand_vocab_2 = {} # key:合并前的token，value:合并后的token

# 加载文本数据
def load_data(path):
    with open(path, 'r', encoding="utf8") as f:
        content = f.read()
        return content

# 合并编码并将新的编码加入到词表当中
def merge(codes, merge_code, code):
    new_codes = list()
    i = 0
    while i < len(codes):
        merge_codes = codes[i:i+2]
        if len(merge_codes) == 2 and (merge_codes[0], merge_codes[1]) == merge_code:
            new_codes.append(code)
            i+=2
        else:
            new_codes.append(merge_codes[0])
            i+=1

    # print(new_codes)
    # print(f'codes length: {len(codes)}, new_codes length: {len(new_codes)}')
    print(f"merging {merge_code} into a new token {code}")
    expand_vocab_1.update({code: merge_code})
    expand_vocab_2.update({merge_code: code})
    return new_codes

# 获取当前需要合并的编码
def get_merge_code(ids):
    statistics = {}
    for j in range(len(ids)-1):
        merge_codes = ids[j:j+2]
        key = (merge_codes[0], merge_codes[1])
        statistics.update({key: statistics.get(key, 0) + 1})

    # print(sorted(((v,k) for k,v in statistics.items()), reverse=True))

    merge_code = max(statistics, key=statistics.get)
    return merge_code


def get_statistics_vocab(ids, num_merges):
    for i in range(num_merges):
        
        merge_code = get_merge_code(ids)
        # print(merge_code)

        ids = merge(ids, merge_code, 256+i)

    # merge([1,2,3,2,3,2,3,4], (2,3), 10)

    print(f'codes length: {len(utf8_codes)}, new_codes length: {len(ids)}')
    print(f"compression ratio: {len(utf8_codes) / len(ids):.2f}X")
    print(f'expand vocab 1: {expand_vocab_1}')
    print(f'expand vocab 2: {expand_vocab_2}')

# 根据新的词表给文本编码
def diy_encode(text:str):
    tokens = list(text.encode("utf-8"))
    i=0
    while i < len(tokens)-1:
        key = (tokens[i], tokens[i+1])
        if key in expand_vocab_2.keys():
            tokens = tokens[0:i] + [expand_vocab_2.get(key)] + tokens[i+2:]
        else:
            i+=1

    return tokens

# 根据新的词表给文本解码
def diy_decode(tokens:list):
    i=0
    while i < len(tokens):
        token = tokens[i]
        if token in expand_vocab_1.keys():
            origin_token = expand_vocab_1.get(token)
            tokens[i:i+1]= [origin_token[0], origin_token[1]]
        else:
            i+=1            

    return tokens


if __name__ == "__main__":
    # 加载文本数据
    text = load_data("./corpus")
    print(text)
    # text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    utf8_codes = list(text.encode("utf-8"))

    ids = list(utf8_codes)
    # # print(utf8_codes)

    vocab_size = 300 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - 256
    # 统计词表
    get_statistics_vocab(ids, num_merges)

    # 测试编码前的字符串与解码后的字符串是否一致
    index = 1
    for origin_str in ["hello world", "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."]:
        # 根据新的词表编码
        encode_tokens = diy_encode(origin_str)
        # print(f'diy_encode: {encode_tokens}')

        # 根据新的词表解码
        decode_tokens = diy_decode(encode_tokens)
        # print(f'diy_decode: {decode_tokens}')

        valid_str = bytes(decode_tokens).decode("utf-8")
        print(f'example {index}: valid result: {origin_str==valid_str}')
        index+=1

    # diy_decode([283, 100, 275, 256])