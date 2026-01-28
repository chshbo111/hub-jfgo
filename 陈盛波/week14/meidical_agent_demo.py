"""
医药行业Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义 ====================
# 每个企业有自己不同的产品，需要企业自己定义

def get_drugs():
    """
    获取所有可用的药品列表
    """
    products = [
        {
            "id": "001",
            "name": "对乙酰氨基酚",
            "type": "西药",
            "indication": "用于感冒发热、关节痛、神经痛及偏头痛、癌性痛及手术后止痛。还可用于对阿司匹林过敏、不耐受或不适于应用阿司匹林的患者（水痘、血友病以及其他出血性疾病等）。",
            "usage": "口服。成人及12岁以上儿童：若症状持续，一次1-2片，每隔4-6小时重复用药一次。24小时内不得超过8片。"
        },
        {
            "id": "002",
            "name": "扑尔敏",
            "type": "西药",
            "indication": "主要用于皮肤黏膜过敏性疾病，如荨麻疹、枯草热、过敏性鼻炎、血管舒张性鼻炎、药疹及湿疹等；亦可用于减轻感冒引起的流涕、打喷嚏、鼻塞等症状。",
            "usage": "口服。成人一次4mg（1片），一日3次。服用后常有嗜睡、口干等副作用，服药期间不得驾驶机、车、船或操作危险机器。"
        },
        {
            "id": "003",
            "name": "999感冒灵颗粒",
            "type": "中成药",
            "indication": "感冒引起的头痛、发热、鼻塞、流涕、咽痛。",
            "usage": "开水冲服。一次1袋，一日3次。注意：含西药成分（对乙酰氨基酚），不可与其他感冒药重叠使用。"
        },
        {
            "id": "004",
            "name": "布洛芬",
            "type": "西药",
            "indication": "普通感冒或流感引起的发热；轻至中度疼痛（如头痛、关节痛、偏头痛、牙痛、痛经）。",
            "usage": "成人：口服，一次1片（0.2g），若持续疼痛或发热，可间隔4-6小时重复用药，24小时内不超过4次。"
        },
        {
            "id": "005",
            "name": "右美沙芬",
            "type": "西药",
            "indication": "用于干咳，包括上呼吸道感染（感冒、咽炎）、支气管炎等引起的咳嗽。",
            "usage": "成人：一次1-2片（15-30mg），一日3次。注意：痰多患者慎用，以免抑制咳嗽反射导致痰液积聚。"
        },
        {
            "id": "006",
            "name": "沐舒坦 (盐酸氨溴索)",
            "type": "西药",
            "indication": "适用于痰液粘稠、不易咳出的急、慢性支气管炎等助力排痰。",
            "usage": "成人：一次1片（30mg），一日3次，饭后服用。"
        },
        {
            "id": "007",
            "name": "开瑞坦 (氯雷他定)",
            "type": "西药",
            "indication": "缓解过敏性鼻炎相关的症状（喷嚏、流涕、鼻痒）；慢性荨麻疹及其他过敏性皮肤病。",
            "usage": "成人及12岁以上儿童：一日1次，一次1片（10mg）。相比扑尔敏，其嗜睡副作用极低。"
        },
        {
            "id": "008",
            "name": "蒙脱石散",
            "type": "西药",
            "indication": "成人及儿童急、慢性腹泻。",
            "usage": "成人：一次1袋（3g），一日3次。服用时需倒入50ml温水中混匀快速服用。注意与其他药物间隔至少1小时。"
        },
        {
            "id": "009",
            "name": "奥美拉唑",
            "type": "西药",
            "indication": "胃溃疡、十二指肠溃疡、应激性溃疡、反流性食管炎及胃灼热（反酸）。",
            "usage": "口服，一次1粒（20mg），一日1-2次。通常建议晨起吞服或睡前服用。"
        },
        {
            "id": "010",
            "name": "雷尼替丁",
            "type": "西药",
            "indication": "用于口服治疗十二指肠溃疡、胃溃疡、反流性食管炎、胃炎等。",
            "usage": "成人：一次150mg（1片），一日2次（早晚各一次），或睡前一次300mg。"
        }
    ]
    return json.dumps(products, ensure_ascii=False)

def get_diseases():
    """
    获取所有可医治的病症列表
    """
    products = [
        {
            "id": "001",
            "name": "普通感冒",
            "symptom": "起病较急，主要表现为鼻塞、流涕、打喷嚏，伴有轻微咽痛、乏力，通常不发热或仅有轻微低热。"
        },
        {
            "id": "002",
            "name": "发烧",
            "symptom": "体温异常升高（腋温≥37.3°C），常伴有面部潮红、皮肤灼热、头痛、肌肉酸痛、畏寒或寒战，心率呼吸加快。"
        },
        {
            "id": "003",
            "name": "干咳 (无痰)",
            "symptom": "咳嗽频繁但无痰液分泌，表现为嗓子发痒、阵发性呛咳，常由过敏、环境刺激或病毒感染后期引起。"
        },
        {
            "id": "004",
            "name": "湿咳 (有痰)",
            "symptom": "咳嗽时伴有明显的痰鸣音，气管分泌物增多，咳出白色粘痰或黄色脓痰，常伴有胸闷感。"
        },
        {
            "id": "005",
            "name": "过敏性鼻炎",
            "symptom": "典型症状为阵发性喷嚏连续发作、大量清水样鼻涕、鼻痒及鼻塞，有时伴有眼痒、流泪等眼部症状。"
        },
        {
            "id": "006",
            "name": "急性胃肠炎",
            "symptom": "主要表现为频繁腹泻（水样便）、腹痛、恶心、呕吐，严重者可出现发热、脱水及电解质紊乱。"
        },
        {
            "id": "007",
            "name": "胃酸/胃痛",
            "symptom": "胸骨后烧灼感（反酸、烧心）、上腹部疼痛或不适，常在进食后或空腹时加重，可能伴有嗳气。"
        }
    ]
    return json.dumps(products, ensure_ascii=False)


# def get_product_detail(product_id: str):
#     """
#     获取指定保险产品的详细信息
    
#     Args:
#         product_id: 产品ID
#     """
#     products = {
#         "life_001": {
#             "id": "life_001",
#             "name": "安心人寿保险",
#             "type": "人寿保险",
#             "description": "为您和家人提供长期保障，身故或全残可获赔付",
#             "coverage": ["身故保障", "全残保障", "重大疾病保障"],
#             "min_amount": 100000,
#             "max_amount": 5000000,
#             "min_years": 10,
#             "max_years": 30,
#             "age_limit": "18-60周岁"
#         },
#         "health_001": {
#             "id": "health_001",
#             "name": "健康无忧医疗险",
#             "type": "医疗保险",
#             "description": "全面覆盖住院、门诊医疗费用",
#             "coverage": ["住院医疗", "门诊医疗", "手术费用", "特殊门诊"],
#             "min_amount": 50000,
#             "max_amount": 1000000,
#             "min_years": 1,
#             "max_years": 5,
#             "age_limit": "0-65周岁"
#         },
#         "accident_001": {
#             "id": "accident_001",
#             "name": "意外伤害保险",
#             "type": "意外险",
#             "description": "保障意外伤害导致的医疗和伤残",
#             "coverage": ["意外身故", "意外伤残", "意外医疗"],
#             "min_amount": 50000,
#             "max_amount": 2000000,
#             "min_years": 1,
#             "max_years": 1,
#             "age_limit": "0-75周岁"
#         }
#     }
    
#     if product_id in products:
#         return json.dumps(products[product_id], ensure_ascii=False)
#     else:
#         return json.dumps({"error": "产品不存在"}, ensure_ascii=False)




# ==================== 工具函数的JSON Schema定义 ====================
# 这部分会成为LLM的提示词的一部分

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_drugs",
            "description": "获取所有可用的药品列表，包括药品名称、类型、适应症、用法用量等基本信息",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_diseases",
            "description": "获取所有可医治的疾病列表，包括疾病名称、症状等基本信息",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]



# ==================== Agent核心逻辑 ====================

# 工具函数映射
available_functions = {
    "get_drugs": get_drugs,
    "get_diseases": get_diseases
}


def run_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    """
    运行Agent，处理用户查询
    
    Args:
        user_query: 用户输入的问题
        api_key: API密钥（如果不提供则从环境变量读取）
        model: 使用的模型名称
    """
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的医生助手。你可以：
1. 介绍各种药品及其详细信息
2. 介绍各种疾病及其详细信息
3. 根据客户的症状描述判断客户得了哪种疾病以及需要吃哪些药品治疗

请根据用户的问题，使用合适的工具来获取信息并给出专业的建议。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    print("\n" + "="*60)
    print("【用户问题】")
    print(user_query)
    print("="*60)
    
    # Agent循环：最多进行5轮工具调用
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")
        
        # 调用大模型
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让模型自主决定是否调用工具
        )
        
        response_message = response.choices[0].message
        
        # 将模型响应加入对话历史
        messages.append(response_message)
        
        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls
        
        if not tool_calls:
            # 没有工具调用，说明模型已经给出最终答案
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("="*60)
            return response_message.content
        
        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")
            
            # 执行对应的函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                
                print(f"工具返回: {function_response[:200]}..." if len(function_response) > 200 else f"工具返回: {function_response}")
                
                # 将工具调用结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                print(f"错误：未找到工具 {function_name}")
    
    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"




# ==================== 示例场景 ====================

def demo_scenarios():
    """
    演示几个典型场景
    """
    print("\n" + "#"*60)
    print("# 保险公司Agent演示 - Function Call能力展示")
    print("#"*60)
    
    # 注意：需要设置环境变量 DASHSCOPE_API_KEY
    # 或者在调用时传入api_key参数
    
    scenarios = [
        "你们有哪些保险产品？",
        "我想了解一下人寿保险的详细信息",
        "我今年35岁，想买50万保额的人寿保险，保20年，需要多少钱？",
        "如果我投保100万的人寿保险30年，到期能有多少收益？",
        "帮我比较一下人寿保险和意外险，保额都是100万，我35岁，保20年"
    ]
    
    print("\n以下是几个示例场景，您可以选择其中一个运行：\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    
    print("\n" + "-"*60)
    print("要运行示例，请取消注释main函数中的相应代码")
    print("并确保设置了环境变量：DASHSCOPE_API_KEY")
    print("-"*60)



if __name__ == "__main__":
    # 展示示例场景
    # demo_scenarios()
    
    # 运行示例（取消注释下面的代码来运行）
    # 注意：需要先设置环境变量 DASHSCOPE_API_KEY
    
    # 示例1：查询药品列表
    # run_agent("你们有哪些药品？", model="qwen-plus")
    
    # 示例2：查询疾病列表
    # run_agent("你们能医治哪些疾病？", model="qwen-plus")
    
    # 示例3：根据症状诊断疾病并给出治疗方案
    # run_agent("我目前的症状是流鼻涕，发烧38.5℃，身体发冷，喉咙也很痛", model="qwen-plus")

    # 自定义查询
    run_agent("我长了一个痔疮", model="qwen-plus")