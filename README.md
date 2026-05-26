<p align="center">
  <img alt="SemViQA Logo" src="https://raw.githubusercontent.com/DAVID-NGUYEN-S16/SemViQA/main/image/logo.png" width="200" /> 
    <br>
  </p>
</p>

# **SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking**  
> 🎉 **News:** SemViQA has been officially accepted to the **ACL 2026 Industry Track**.

### **Authors**:  
[**Dien X. Tran**](https://github.com/xndien2004)\*, [**Nam V. Nguyen**](https://github.com/DAVID-NGUYEN-S16)\*, Thanh T. Tran, Anh T. Hoang, Tai V. Duong, Di T. Le, Phuc-Lu Le 
<p align="center">
  <a href="https://arxiv.org/abs/2503.00955">
    <img src="https://img.shields.io/badge/arXiv-2411.00918-red?style=flat&label=arXiv">
  </a>
  <a href="https://huggingface.co/SemViQA">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow?style=flat">
  </a>
  <a href="https://pypi.org/project/SemViQA">
    <img src="https://img.shields.io/pypi/v/SemViQA?color=blue&label=PyPI">
  </a>
  <a href="https://github.com/DAVID-NGUYEN-S16/SemViQA">
    <img src="https://img.shields.io/github/stars/DAVID-NGUYEN-S16/SemViQA?style=social">
  </a>
</p>


<p align="center">
    <a href="#-about">📌 About</a> •
    <a href="#-checkpoints">🔍 Checkpoints</a> •
    <a href="#-quick-start">🚀 Quick Start</a> •
    <a href="#-training">🏋️‍♂️ Training</a> •
    <a href="#-evaluation">📊 Evaluation</a> •
    <a href="#-citation">📖 Citation</a>
</p>  

---

## 📌 **About**  

Misinformation is a growing problem, exacerbated by the increasing use of **Large Language Models (LLMs)** like GPT and Gemini. This issue is even more critical for **low-resource languages like Vietnamese**, where existing fact-checking methods struggle with **semantic ambiguity, homonyms, and complex linguistic structures**.  

To address these challenges, we introduce **SemViQA**, a novel **Vietnamese fact-checking framework** integrating:  

- **Semantic-based Evidence Retrieval (SER)**: Combines **TF-IDF** with a **Question Answering Token Classifier (QATC)** to enhance retrieval precision while reducing inference time.  
- **Two-step Verdict Classification (TVC)**: Uses hierarchical classification optimized with **Cross-Entropy and Focal Loss**, improving claim verification across three categories:  
  - **Supported** ✅  
  - **Refuted** ❌  
  - **Not Enough Information (NEI)** 🤷‍♂️  

### **🏆 Achievements**
- **1st place** in the **UIT Data Science Challenge** 🏅  
- **State-of-the-art** performance on:  
  - **ISE-DSC01** → **78.97% strict accuracy**  
  - **ViWikiFC** → **80.82% strict accuracy**  
- **SemViQA Faster**: **7x speed improvement** over the standard model 🚀  

These results establish **SemViQA** as a **benchmark for Vietnamese fact verification**, advancing efforts to combat misinformation and ensure **information integrity**.  

---
## 🔍 Checkpoints
We are making our **SemViQA** experiment checkpoints publicly available to support the **Vietnamese fact-checking research community**. By sharing these models, we aim to:  

- **Facilitate reproducibility**: Allow researchers and developers to validate and build upon our results.  
- **Save computational resources**: Enable fine-tuning or transfer learning on top of **pre-trained and fine-tuned models** instead of training from scratch.  
- **Encourage further improvements**: Provide a strong baseline for future advancements in **Vietnamese misinformation detection**.  
 

<table>
  <tr>
    <th>Method</th>
    <th>Model</th>
    <th>ViWikiFC</th>
    <th>ISE-DSC01</th>
  </tr>
  <tr>
    <td rowspan="3"><strong>TC</strong></td>
    <td>InfoXLM<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/tc-infoxlm-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/tc-infoxlm-isedsc01">Link</a></td>
  </tr>
  <tr>
    <td>XLM-R<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/tc-xlmr-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/tc-xlmr-isedsc01">Link</a></td>
  </tr>
  <tr>
    <td>Ernie-M<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/tc-erniem-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/tc-erniem-isedsc01">Link</a></td> 
  </tr>
  <tr>
    <td rowspan="3"><strong>BC</strong></td>
    <td>InfoXLM<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/bc-infoxlm-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/bc-infoxlm-isedsc01">Link</a></td>
  </tr>
  <tr>
    <td>XLM-R<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/bc-xlmr-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/bc-xlmr-isedsc01">Link</a></td>
  </tr>
  <tr>
    <td>Ernie-M<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/bc-erniem-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/bc-erniem-isedsc01">Link</a></td>
  </tr>
  <tr>
    <td rowspan="2"><strong>QATC</strong></td>
    <td>InfoXLM<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/qatc-infoxlm-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/qatc-infoxlm-isedsc01">Link</a></td>
  </tr>
  <tr>
    <td>ViMRC<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/qatc-vimrc-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/qatc-vimrc-isedsc01">Link</a></td>
  </tr>
  <tr>
    <td rowspan="2"><strong>QA origin</strong></td>
    <td>InfoXLM<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/infoxlm-large-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/infoxlm-large-isedsc01">Link</a></td>
  </tr>
  <tr>
    <td>ViMRC<sub>large</sub></td>
    <td><a href="https://huggingface.co/SemViQA/vi-mrc-large-viwikifc">Link</a></td>
    <td><a href="https://huggingface.co/SemViQA/vi-mrc-large-isedsc01">Link</a></td>
  </tr>
</table>

 

---

## 🚀 **Quick Start**  

### 📥 **Installation**  

#### **1️⃣ Clone this repository**  
```bash
git clone https://github.com/DAVID-NGUYEN-S16/SemViQA.git
cd SemViQA
```

#### **2️⃣ Set up Python environment**  
We recommend using **Python 3.11** in a virtual environment (`venv`) or **Anaconda**.  

**Using `venv`:**  
```bash
python -m venv semviqa_env
source semviqa_env/bin/activate  # On MacOS/Linux
semviqa_env\Scripts\activate      # On Windows
```

**Using `Anaconda`:**  
```bash
conda create -n semviqa_env python=3.11 -y
conda activate semviqa_env
```

#### **3️⃣ Install dependencies**  
```bash
pip install --upgrade pip
pip install transformers==4.42.3
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
---

## 🏋️‍♂️ **Training**  

Train different components of **SemViQA** using the provided scripts:  

### **1️⃣ Three-Class Classification Training**  
Train a three-class claim classification model using the following command:
```bash
bash scripts/tc.sh
```
If you want to fine-tune the model using pre-trained weights, you can initialize it as follows:
```python
# Install semviqa
!pip install semviqa

# Initalize a pipeline
from transformers import AutoTokenizer
from semviqa.tvc.model import ClaimModelForClassification

tokenizer = AutoTokenizer.from_pretrained("SemViQA/tc-infoxlm-viwikifc")
model = ClaimModelForClassification.from_pretrained("SemViQA/tc-infoxlm-viwikifc", num_labels=3)
```

### **2️⃣ Binary Classification Training**  
Train a binary classification model using the command below:
```bash
bash scripts/bc.sh
```
To fine-tune the model with existing weights, use the following setup:
```python
# Install semviqa
!pip install semviqa

# Initalize a pipeline
from transformers import AutoTokenizer
from semviqa.tvc.model import ClaimModelForClassification

tokenizer = AutoTokenizer.from_pretrained("SemViQA/bc-infoxlm-viwikifc")
model = ClaimModelForClassification.from_pretrained("SemViQA/bc-infoxlm-viwikifc", num_labels=2)
```

### **3️⃣ QATC Model Training**  
Train the Question Answering Token Classifier (QATC) model using the following command:
```bash
bash scripts/qatc.sh
```

To continue training from pre-trained weights, use this setup:
```python
# Install semviqa
!pip install semviqa

# Initalize a pipeline
from transformers import AutoTokenizer
from semviqa.ser.qatc_model import QATCForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("SemViQA/qatc-infoxlm-viwikifc")
model = QATCForQuestionAnswering.from_pretrained("SemViQA/qatc-infoxlm-viwikifc")
```

---

## 📊 **Evaluation**

### **1️⃣ Semantic-based Evidence Retrieval**
This module extracts the most relevant evidence from a given context based on a claim. It leverages TF-IDF combined with the QATC model to ensure accurate retrieval.
```python
# Install semviqa package
!pip install semviqa

# Import the ser module
import torch
from transformers import AutoTokenizer
from semviqa.ser.qatc_model import QATCForQuestionAnswering
from semviqa.ser.ser_eval import extract_evidence_tfidf_qatc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("SemViQA/qatc-infoxlm-viwikifc")
model = QATCForQuestionAnswering.from_pretrained("SemViQA/qatc-infoxlm-viwikifc")

claim = "Chiến tranh với Campuchia đã kết thúc trước khi Việt Nam thống nhất."
context = "Sau khi thống nhất, Việt Nam tiếp tục gặp khó khăn do sự sụp đổ và tan rã của đồng minh Liên Xô cùng Khối phía Đông, các lệnh cấm vận của Hoa Kỳ, chiến tranh với Campuchia, biên giới giáp Trung Quốc và hậu quả của chính sách bao cấp sau nhiều năm áp dụng. Năm 1986, Đảng Cộng sản ban hành cải cách đổi mới, tạo điều kiện hình thành kinh tế thị trường và hội nhập sâu rộng. Cải cách đổi mới kết hợp cùng quy mô dân số lớn đưa Việt Nam trở thành một trong những nước đang phát triển có tốc độ tăng trưởng thuộc nhóm nhanh nhất thế giới, được coi là Hổ mới châu Á dù cho vẫn gặp phải những thách thức như tham nhũng, tội phạm gia tăng, ô nhiễm môi trường và phúc lợi xã hội chưa đầy đủ. Ngoài ra, giới bất đồng chính kiến, chính phủ một số nước phương Tây và các tổ chức theo dõi nhân quyền có quan điểm chỉ trích hồ sơ nhân quyền của Việt Nam liên quan đến các vấn đề tôn giáo, kiểm duyệt truyền thông, hạn chế hoạt động ủng hộ nhân quyền cùng các quyền tự do dân sự."

evidence = extract_evidence_tfidf_qatc(
    claim, context, model, tokenizer, device, confidence_threshold=0.5, length_ratio_threshold=0.6
)

print(evidence)
# evidence: sau khi thống nhất việt nam tiếp tục gặp khó khăn do sự sụp đổ và tan rã của đồng minh liên xô cùng khối phía đông các lệnh cấm vận của hoa kỳ chiến tranh với campuchia biên giới giáp trung quốc và hậu quả của chính sách bao cấp sau nhiều năm áp dụng
```


### **2️⃣ Two-step Verdict Classification**
This module performs claim classification using a **two-step approach**:
1. **Three-class classification**: Determines if a claim is **Supported, Refuted, or Not Enough Information (NEI)**.
2. **Binary classification** (if necessary): Further verifies if the claim is **Supported** or **Refuted**.
```python
# Install semviqa package
!pip install semviqa

# Import the tvc module
import torch
from semviqa.tvc.tvc_eval import classify_claim
from semviqa.tvc.model import ClaimModelForClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("SemViQA/tc-infoxlm-viwikifc")
model_tc = ClaimModelForClassification.from_pretrained("SemViQA/tc-infoxlm-viwikifc", num_labels=3).to(device)
model_bc = ClaimModelForClassification.from_pretrained("SemViQA/bc-infoxlm-viwikifc", num_labels=2).to(device)

claim = "Chiến tranh với Campuchia đã kết thúc trước khi Việt Nam thống nhất."
evidence = "Sau khi thống nhất, Việt Nam tiếp tục gặp khó khăn do sự sụp đổ và tan rã của đồng minh Liên Xô cùng Khối phía Đông, các lệnh cấm vận của Hoa Kỳ, chiến tranh với Campuchia, biên giới giáp Trung Quốc và hậu quả của chính sách bao cấp sau nhiều năm áp dụng."

verdict = "NEI"
prob_tc, pred_tc = classify_claim(claim, evidence, model_tc, tokenizer, device)

if pred_tc != 0:
  prob_bc, pred_bc = classify_claim(claim, evidence, model_bc, tokenizer, device)
  verdict = "SUPPORTED" if pred_bc == 0 else "REFUTED" if prob_bc > prob_tc else ["NEI", "SUPPORTED", "REFUTED"][pred_tc]

print(verdict)
# Output: REFUTED
```

### **3️⃣ Full Pipeline Evaluation**
Use the trained models to **predict test data**:  
```bash
bash scripts/pipeline.sh
```

Alternatively, you can use **SemViQA** programmatically:

```python
# Install semviqa package
!pip install semviqa

# Import the pipeline
from semviqa.pipeline import SemViQAPipeline
claim = "Chiến tranh với Campuchia đã kết thúc trước khi Việt Nam thống nhất."
context = "Sau khi thống nhất, Việt Nam tiếp tục gặp khó khăn do sự sụp đổ và tan rã của đồng minh Liên Xô cùng Khối phía Đông, các lệnh cấm vận của Hoa Kỳ, chiến tranh với Campuchia, biên giới giáp Trung Quốc và hậu quả của chính sách bao cấp sau nhiều năm áp dụng. Năm 1986, Đảng Cộng sản ban hành cải cách đổi mới, tạo điều kiện hình thành kinh tế thị trường và hội nhập sâu rộng. Cải cách đổi mới kết hợp cùng quy mô dân số lớn đưa Việt Nam trở thành một trong những nước đang phát triển có tốc độ tăng trưởng thuộc nhóm nhanh nhất thế giới, được coi là Hổ mới châu Á dù cho vẫn gặp phải những thách thức như tham nhũng, tội phạm gia tăng, ô nhiễm môi trường và phúc lợi xã hội chưa đầy đủ. Ngoài ra, giới bất đồng chính kiến, chính phủ một số nước phương Tây và các tổ chức theo dõi nhân quyền có quan điểm chỉ trích hồ sơ nhân quyền của Việt Nam liên quan đến các vấn đề tôn giáo, kiểm duyệt truyền thông, hạn chế hoạt động ủng hộ nhân quyền cùng các quyền tự do dân sự."
 
semviqa = SemViQAPipeline(
  model_evidence_QA="SemViQA/qatc-infoxlm-viwikifc", 
  model_bc="SemViQA/bc-infoxlm-viwikifc", 
  model_tc="SemViQA/tc-infoxlm-viwikifc", 
  thres_evidence=0.5,
  length_ratio_threshold=0.5,
  is_qatc_faster=False
  )
 
result = semviqa.predict(claim, context)
print(result)
# Output: {'verdict': 'REFUTED', 'evidence': 'sau khi thống nhất việt nam tiếp tục gặp khó khăn do sự sụp đổ và tan rã của đồng minh liên xô cùng khối phía đông các lệnh cấm vận của hoa kỳ chiến tranh với campuchia biên giới giáp trung quốc và hậu quả của chính sách bao cấp sau nhiều năm áp dụng'}

# Extract only evidence
evidence_only = semviqa.predict(claim, context, return_evidence_only=True)
print(evidence_only)
# Output: {'evidence': 'sau khi thống nhất việt nam tiếp tục gặp khó khăn do sự sụp đổ và tan rã của đồng minh liên xô cùng khối phía đông các lệnh cấm vận của hoa kỳ chiến tranh với campuchia biên giới giáp trung quốc và hậu quả của chính sách bao cấp sau nhiều năm áp dụng'}
```

## **Acknowledgment**  
This work builds upon prior research by Nam V. Nguyen, who served as co-first author of the SemViQA paper.
- [Check-Fact-Question-Answering-System](https://github.com/DAVID-NGUYEN-S16/Check-Fact-Question-Answering-System)  
- [Extract-Evidence-Question-Answering](https://github.com/DAVID-NGUYEN-S16/Extract-evidence-question-answering)  

**SemViQA** is the final version we have developed for verifying fact-checking in Vietnamese, achieving state-of-the-art (SOTA) performance compared to any other system for Vietnamese.

## 📖 **Citation**  

If you use **SemViQA** in your research, please cite our work:  

```bibtex
@misc{tran2025semviqasemanticquestionanswering,
      title={SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking}, 
      author={Dien X. Tran and Nam V. Nguyen and Thanh T. Tran and Anh T. Hoang and Tai V. Duong and Di T. Le and Phuc-Lu Le},
      year={2025},
      eprint={2503.00955},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.00955}, 
}
```  
