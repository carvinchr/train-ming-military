"""
评估 Step 1: 创建人工标注测试集模板
生成后需要人工填写 reference_answer 字段
用法：python 04_evaluation/create_testset.py --output data/testset.jsonl
"""
import argparse
import json
from pathlib import Path

# 精心设计的测试问题（覆盖不同难度和类型）
TEST_QUESTIONS = [
    # 基础事实类
    {"id": "q01", "category": "事实", "difficulty": "easy",
     "question": "明代神机营是何时建立的？其主要职能是什么？",
     "reference_answer": ""},

    {"id": "q02", "category": "事实", "difficulty": "easy",
     "question": "佛朗机铳最早是通过什么途径传入明朝的？",
     "reference_answer": ""},

    {"id": "q03", "category": "事实", "difficulty": "easy",
     "question": "明代虎蹲炮的主要特征和用途是什么？",
     "reference_answer": ""},

    {"id": "q04", "category": "事实", "difficulty": "medium",
     "question": "红夷大炮与明代本土铸炮在制造工艺上有何主要区别？",
     "reference_answer": ""},

    {"id": "q05", "category": "事实", "difficulty": "medium",
     "question": "明军三眼铳的结构特点和战术运用方式是什么？",
     "reference_answer": ""},

    # 比较分析类
    {"id": "q06", "category": "比较", "difficulty": "medium",
     "question": "明代火绳枪（鸟铳）与欧洲同时期的火绳枪相比，在性能和工艺上有何异同？",
     "reference_answer": ""},

    {"id": "q07", "category": "比较", "difficulty": "hard",
     "question": "明初与明末火器技术的发展水平有何变化？主要原因是什么？",
     "reference_answer": ""},

    {"id": "q08", "category": "比较", "difficulty": "hard",
     "question": "戚继光与孙承宗在军事装备体系建设上的理念有何异同？",
     "reference_answer": ""},

    # 因果分析类
    {"id": "q09", "category": "因果", "difficulty": "medium",
     "question": "明代为何在引进欧洲火炮技术后，仍然无法在辽东战场占据优势？",
     "reference_answer": ""},

    {"id": "q10", "category": "因果", "difficulty": "hard",
     "question": "明代车营战术的兴起与当时火器发展水平有何内在关联？",
     "reference_answer": ""},

    # 综合论述类
    {"id": "q11", "category": "论述", "difficulty": "hard",
     "question": "请综合评述明代火器发展的成就与局限，以及其对明朝兴衰的影响。",
     "reference_answer": ""},

    {"id": "q12", "category": "论述", "difficulty": "hard",
     "question": "明代水军的战船配置和火力体系是如何演变的？",
     "reference_answer": ""},

    # 细节考证类
    {"id": "q13", "category": "考证", "difficulty": "hard",
     "question": "《武备志》中记载的连珠炮是一种什么样的武器？其技术原理如何？",
     "reference_answer": ""},

    {"id": "q14", "category": "考证", "difficulty": "medium",
     "question": "明代铠甲的主要种类有哪些？各自的材料和防护效果如何？",
     "reference_answer": ""},

    {"id": "q15", "category": "考证", "difficulty": "easy",
     "question": "明代的骑兵弓箭与步兵弓箭在规格和用法上有何区别？",
     "reference_answer": ""},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/testset.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for q in TEST_QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"✅ 生成测试集模板: {output_path}（{len(TEST_QUESTIONS)} 题）")
    print(f"\n请用文本编辑器打开 {output_path}")
    print("为每题的 reference_answer 字段填写标准参考答案，然后再运行评估脚本。")
    print("\n题目分类统计：")
    from collections import Counter
    cats = Counter(q["category"] for q in TEST_QUESTIONS)
    for cat, count in cats.items():
        print(f"  {cat}: {count} 题")


if __name__ == "__main__":
    main()
