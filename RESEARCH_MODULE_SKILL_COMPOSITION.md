# Research — 模块拆分成 Skill 后，如何与 Brain 可靠串联？

> 调研问题（学长 Task 3）：如果把上面的"单一微调 Qwen 承担所有角色"拆成不同模块（Skill），不同 Skill 之间如何协作？如何与 Brain（大脑）可靠串起来？
>
> 一句话答案：**需要一个"统一 Skill 表示（USR）+ 强校验通道（SkillChannel）+ 对齐层（Adapter）"作为 Skill 与 Brain 之间的中间语言**——Brain 只消费 USR 的公共字段（object.class/found/confidence），Skill 通过 publish/consume 交互，Adapter 负责把异构 FM 输出对齐到 Brain 期望的契约。

---

## 1. 为什么直接拆开会崩

RoboAgent 把 7 个角色耦合在**同一个微调权重**里。如果拆成独立 Skill（外部 FM），会遇到：

1. **Contract Mismatch**：外部 FM 输出的 label/格式与微调 Brain 期望的词表/契约不匹配
   - 实测：GDINO 直接替换 OG → EB SR 从 78% 崩到 34%（-44pp）
2. **无统一接口**：每个 Skill 的输出格式不同（检测框 vs 描述 vs 方向），Brain 无法统一消费
3. **Brain 不消费信号**：即使 Skill 输出置信度/不确定性，微调 Brain 也不理会（反事实 0/35 sensitive）

---

## 2. 我们的方案：三层桥接

```
外部 FM (Skill) → [Adapter 对齐] → USR (统一表示) → [SkillChannel 校验] → Brain
```

### Layer 1 — Adapter（Decision-Compatible Adapter）
- 把异构 FM 输出对齐到 Brain 期望契约：canonical 词表 + functional override + no-veto + found/fallback
- 效果：OG 替换从 34% → 80%（+46pp）

### Layer 2 — USR（Unified Skill Representation）
- 统一 Skill 输出为结构化表示：`environment_facts / task_semantics / decision_signals / provenance`
- 公共字段：object.class / found / confidence / relation / location
- 模型私有字段（det_query/bbox/raw）剥离，不进入公共表示

### Layer 3 — SkillChannel（校验通道）
- `publish(skill, USR)` / `consume(skill, public_field)` 强制
- 白名单校验（PUBLIC_FIELDS）+ 跨 episode/step 隔离 + contract audit
- 阻止 raw 信息绕过通道进入下游

---

## 3. Skill 间如何串联（传播链）

实测验证的传播路径（无断链）：

```
Path A: OG found=false → EG/reobserve（10/10 下一 skill 是 EG）
Path B: OG object.class → SD → LPM → action（SD input == OG class 3/3）
Path C: USR found/confidence → Decision-aware Brain → execute/guard/reobserve（34/35）
```

关键：**每个 Skill 只消费 USR 的公共字段，不直接读其他 Skill 的原始输出**。

---

## 4. 为什么需要 Decision-aware Training

即使有了 USR，**baseline Brain 不会消费 decision signals**（反事实 0/35）。必须额外训练 Brain 学会：

| 信号状态 | 期望行为 | DA-FT 实测 |
|----------|---------|-----------|
| high-confidence | execute | ✅ |
| low-confidence | guard (no-op) | ✅ |
| found=false | reobserve | ✅ |
| conflict | reobserve | ✅ |

DA-FT decision accuracy 98%，signal-sensitivity 34/35。

---

## 5. 结论

"拆成模块后如何与 Brain 串起来"的答案：

1. **对齐**（Adapter）：让异构 FM 输出与 Brain 契约兼容
2. **统一表示**（USR）：让 Skill 之间、Skill 与 Brain 之间有共同语言
3. **强校验**（SkillChannel）：保证信息流干净、可审计、无 raw 绕过
4. **训练**（Decision-aware）：让 Brain 学会消费信号，实现信号条件化行为

这四步把"单一权重耦合"变成"可插拔 Skill + 可靠接口"的框架。
