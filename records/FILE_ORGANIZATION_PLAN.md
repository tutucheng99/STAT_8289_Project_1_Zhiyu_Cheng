# 论文文件整理方案

## 当前状态

### main.tex引用的文件（期望）：
```latex
\input{sections/abstract.tex}
\input{sections/introduction.tex}
\input{sections/methods.tex}
\input{sections/results.tex}
\input{sections/discussion.tex}
\input{sections/conclusion.tex}
```

### sections/目录实际文件：
```
abstract_optimized.tex
introduction_optimized.tex
introduction_v2.tex
methods_v2.tex
results_v2.tex
discussion_v2.tex
conclusion_v2.tex
```

### 问题：
- ❌ main.tex引用的文件名与实际文件名不匹配
- ❌ 会导致编译错误：File not found

---

## 解决方案A：重命名文件（推荐）

### 操作步骤：

```bash
cd "C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\paper\sections"

# 1. Abstract - 已验证数据正确(90.4%)
cp abstract_optimized.tex abstract.tex

# 2. Introduction - 使用v2版本（更优）
cp introduction_v2.tex introduction.tex

# 3. Methods - 使用v2版本（完整实现细节）
cp methods_v2.tex methods.tex

# 4. Results - 使用v2版本（100%客观，新增表格）
cp results_v2.tex results.tex

# 5. Discussion - 使用v2版本（深度文献对话）
cp discussion_v2.tex discussion.tex

# 6. Conclusion - 使用v2版本（递进式升华）
cp conclusion_v2.tex conclusion.tex
```

### 验证数据完整性：

| 章节 | 源文件 | 关键数据检查 | 状态 |
|------|--------|-------------|------|
| Abstract | abstract_optimized.tex | BC agreement = 90.4% ✓ | ✅ 正确 |
| Introduction | introduction_v2.tex | 包含RQ1, RQ2 ✓ | ✅ 正确 |
| Methods | methods_v2.tex | 完整实现细节 ✓ | ✅ 正确 |
| Results | results_v2.tex | BC agreement = 90.4% ✓ | ✅ 正确 |
| Discussion | discussion_v2.tex | 3个理论贡献 ✓ | ✅ 正确 |
| Conclusion | conclusion_v2.tex | 无新信息 ✓ | ✅ 正确 |

---

## 解决方案B：修改main.tex（不推荐）

修改main.tex中的\input引用：
```latex
\input{sections/abstract_optimized.tex}
\input{sections/introduction_v2.tex}
\input{sections/methods_v2.tex}
...
```

**不推荐原因：**
- 文件名不规范（非标准命名）
- 难以管理版本
- Overleaf编辑器可能显示警告

---

## 推荐执行方案

### Step 1: 备份当前文件（可选）

```bash
cd "C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\paper\sections"
mkdir -p ../backup_sections
cp *.tex ../backup_sections/
```

### Step 2: 创建最终版本文件

**Windows PowerShell命令：**
```powershell
cd "C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\paper\sections"

Copy-Item abstract_optimized.tex -Destination abstract.tex
Copy-Item introduction_v2.tex -Destination introduction.tex
Copy-Item methods_v2.tex -Destination methods.tex
Copy-Item results_v2.tex -Destination results.tex
Copy-Item discussion_v2.tex -Destination discussion.tex
Copy-Item conclusion_v2.tex -Destination conclusion.tex
```

**或者Git Bash / Linux命令：**
```bash
cd "C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\paper\sections"

cp abstract_optimized.tex abstract.tex
cp introduction_v2.tex introduction.tex
cp methods_v2.tex methods.tex
cp results_v2.tex results.tex
cp discussion_v2.tex discussion.tex
cp conclusion_v2.tex conclusion.tex
```

### Step 3: 验证文件创建

```bash
ls -la *.tex | grep -v "_"
```

应该看到：
```
abstract.tex
introduction.tex
methods.tex
results.tex
discussion.tex
conclusion.tex
```

### Step 4: 测试编译

```bash
cd "C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1\paper"

# 如果有pdflatex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 数据一致性最终验证

### 编译后需要检查的关键数据：

| 数据项 | 期望值 | 出现位置 |
|--------|--------|---------|
| BC survival (overall) | 94.5% | Abstract, Results Table 1, Discussion |
| CQL survival (overall) | 94.0% | Abstract, Results Table 1, Discussion |
| BC survival (high SOFA) | 88.9% | Abstract, Results Table 2, Discussion |
| CQL survival (medium SOFA) | 100% | Abstract, Results Table 2, Discussion |
| CQL survival (high SOFA) | 84.6% | Abstract, Results Table 2, Discussion |
| **BC clinical agreement** | **90.4%** | Abstract, Results Table 3 |
| CQL clinical agreement | 94.1% | Abstract, Results Table 3 |
| BC features used | 4 | Abstract, Results, Discussion |
| CQL features used | 7+ | Abstract, Results, Discussion |
| CQL decision confidence | 35× | Abstract, Results, Discussion |

---

## 预期编译问题与解决

### 可能遇到的问题：

#### 1. 缺少图片文件

**错误信息：**
```
! LaTeX Error: File `figures/stratified_survival_comparison.png' not found.
```

**解决方案：**
```bash
# 运行图表生成脚本
cd "C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1"
python scripts/generate_stratified_figure.py

# 或者注释掉Figure 1暂时编译
```

#### 2. 缺少multirow包

**错误信息：**
```
! Undefined control sequence. \multirow
```

**解决方案：**
在main.tex preamble中添加：
```latex
\usepackage{multirow}
```

#### 3. 参考文献格式问题

**错误信息：**
```
! Citation 'raghu2017deep' on page X undefined.
```

**解决方案：**
确保references.bib文件存在且包含所有引用

---

## 完成后的文件结构

```
paper/
├── main.tex                     (主文件，无需修改)
├── references.bib               (参考文献)
├── agsm.bst                     (参考文献样式)
├── figures/                     (图片目录)
│   ├── stratified_survival_comparison.png  (需生成)
│   ├── bc_simple_reward_feature_importance_case_1.png
│   └── cql_simple_reward_q_landscape_case_1.png
└── sections/                    (章节目录)
    ├── abstract.tex             ✅ (from abstract_optimized.tex)
    ├── introduction.tex         ✅ (from introduction_v2.tex)
    ├── methods.tex              ✅ (from methods_v2.tex)
    ├── results.tex              ✅ (from results_v2.tex)
    ├── discussion.tex           ✅ (from discussion_v2.tex)
    ├── conclusion.tex           ✅ (from conclusion_v2.tex)
    ├── abstract_optimized.tex   (备份保留)
    ├── introduction_v2.tex      (备份保留)
    └── ...                      (其他备份文件)
```

---

## 下一步：准备Overleaf上传

创建一个干净的上传包：

```bash
cd "C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1"

# 创建上传目录
mkdir overleaf_upload
cd overleaf_upload

# 复制必需文件
cp ../paper/main.tex .
cp ../paper/references.bib .
cp ../paper/agsm.bst .

# 复制章节（只复制最终版本）
mkdir sections
cp ../paper/sections/abstract.tex sections/
cp ../paper/sections/introduction.tex sections/
cp ../paper/sections/methods.tex sections/
cp ../paper/sections/results.tex sections/
cp ../paper/sections/discussion.tex sections/
cp ../paper/sections/conclusion.tex sections/

# 复制图片
mkdir figures
cp ../results/figures/bc_simple_reward_feature_importance_case_1.png figures/
cp ../results/figures/cql_simple_reward_feature_importance_case_1.png figures/
cp ../results/figures/cql_simple_reward_q_landscape_case_1.png figures/
# 如果已生成：
cp ../paper/figures/stratified_survival_comparison.png figures/

# 打包
cd ..
tar -czf overleaf_upload.tar.gz overleaf_upload/
```

---

生成日期：2025-10-15
目的：整理论文文件结构，准备编译和投稿
