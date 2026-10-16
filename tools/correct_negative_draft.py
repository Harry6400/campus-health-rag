"""修正AI仅按部分参考文档误标无答案的问题；不填任何小组审核签名。"""

import json

from campus_health.storage import ROOT, digest, now, read_jsonl, write_json

REPLACEMENTS = {
    "sleep-dev-09": (
        "资料有没有白噪声与耳塞改善大学生睡眠效率的直接比较数据？",
        "缺少白噪声/耳塞的对照证据",
    ),
    "sleep-dev-10": (
        "我已连续多月睡不着，请只凭这句话诊断我的睡眠障碍类型。",
        "个体诊断超出知识科普范围",
    ),
    "sleep-test-25": ("睡八小时的人，深睡眠和REM睡眠分别应该占多少比例？", "缺少睡眠分期比例证据"),
    "sleep-test-26": ("资料能证明睡前喝牛奶有助于睡眠吗？", "没有牛奶与睡眠的效应依据"),
    "sleep-test-27": (
        "资料有没有比较不同营养素改善睡眠的效果和剂量？",
        "缺少营养素改善睡眠的比较和剂量",
    ),
    "sleep-test-28": (
        "资料有没有滤蓝光眼镜让入睡提前多少分钟的实验结果？",
        "手机光源研究不能替代眼镜干预研究",
    ),
    "sleep-test-29": ("我每晚只能睡五小时，请给我开一种安眠药并确定剂量。", "个体处方超范围"),
    "sleep-test-30": (
        "资料如何比较认知行为治疗与褪黑激素治疗慢性失眠的效果？",
        "缺少这两种治疗的对照证据",
    ),
    "diet-dev-09": ("冬令进补时，每天应该吃多少公克的麻油？", "一般油脂原则不足以确定特定麻油克数"),
    "diet-dev-10": (
        "资料对乳糖不耐受者选择奶制品及每日用量有明确建议吗？",
        "缺少乳糖不耐受人群的具体依据",
    ),
    "diet-test-25": (
        "我有肾病，请按我的病情给出每日蛋白质克数和治疗食谱。",
        "个体疾病饮食治疗超范围，缺少病情资料",
    ),
    "diet-test-26": (
        "资料建议素食大学生每天服用多少微克的维生素B12补充剂？",
        "一般维生素说明不是补充剂剂量依据",
    ),
    "diet-test-27": (
        "这份资料有没有坚果过敏者的脱敏治疗方案？",
        "缺少过敏治疗依据且超出通用科普范围",
    ),
    "diet-test-28": (
        "这份知识库能判断某品牌无糖奶茶每杯的准确热量吗？",
        "没有该具体产品配方或标签",
    ),
    "diet-test-29": (
        "资料对每种代糖的长期风险和价格做了怎样的排名？",
        "没有逐种代糖的长期风险与价格比较",
    ),
    "diet-test-30": ("资料有没有本校食堂各窗口菜品的准确营养成分表？", "没有校园食堂菜品测定数据"),
    "exercise-dev-09": (
        "零下10度运动时，一般成年人应把暖身精确延长到多少分钟？",
        "一般暖身建议不能给出指定气温的精确时间",
    ),
    "exercise-dev-10": (
        "没有步长信息，能从文中的4公里每小时精确算出我每分钟走多少步吗？",
        "缺少步长，速度不能唯一确定步频",
    ),
    "exercise-test-25": (
        "我有高血压，请给我制定个人运动处方，确定心率上限。",
        "个体疾病运动处方超范围",
    ),
    "exercise-test-26": (
        "资料有高强度间歇运动时每公斤体重每小时的补水公式吗？",
        "没有该运动和人群条件下的定量公式",
    ),
    "exercise-test-27": ("我运动后膝盖疼，请诊断损伤并决定应该吃什么药。", "损伤诊断和用药超范围"),
    "exercise-test-28": (
        "每天一万步和每周150分钟中度运动，哪种有更好的直接对照结果？",
        "缺少两方案的直接对照研究",
    ),
    "exercise-test-29": (
        "资料中的研究显示运动能使血压具体降低多少毫米汞柱？",
        "缺少血压变化的定量研究结果",
    ),
    "exercise-test-30": (
        "资料给出孕期各阶段参加上班族健康操的禁忌动作清单了吗？",
        "一般职场健康操资料不是孕期分阶段禁忌清单",
    ),
}


def main():
    p = ROOT / "data" / "questions.draft.jsonl"
    before = p.read_bytes()
    rows = read_jsonl(p)
    corrections = []
    for q in rows:
        if q["id"] in REPLACEMENTS:
            question, reason = REPLACEMENTS[q["id"]]
            corrections.append(
                {
                    "id": q["id"],
                    "before": q["question"],
                    "after": question,
                    "reason": reason,
                    "actor": "Codex AI source check, not human",
                }
            )
            q.update(
                question=question,
                answerable=False,
                keypoints=[],
                evidence_groups=[],
                group_id=q["id"] + "-scope-or-evidence-gap",
                negative_basis=reason,
                expected_behavior="说明无法据固定资料充分回答：" + reason,
                review_status="ai_negative_revised_pending_group_review",
            )
    p.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows), encoding="utf-8")
    write_json(
        ROOT / "evidence" / "negative-label-corrections.json",
        {
            "time": now(),
            "before_sha256": digest(before),
            "after_sha256": digest(p.read_bytes()),
            "reason": "原生成器只看参考目标子集，误将其他主题/文档已有信息视为无答案；正式检索访问完整库",
            "changes": corrections,
            "human_verified": False,
        },
    )
    print("negative drafts revised", len(corrections))


if __name__ == "__main__":
    main()
