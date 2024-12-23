from mistune import Markdown
import torch
from transformers import AutoTokenizer, BertModel
from gradio import Interface, Textbox
from torch import nn

# 定义模型类
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained('/home/zlz/CodeWithDataset/bert-base-uncased', mirror='tuna')
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, src):
        outputs = self.bert(**src).last_hidden_state[:, 0, :]
        return self.predictor(outputs)

# 加载训练好的模型
model_path = "/home/zlz/model_rec/model/bert_checkpoints/model_best.pt"
model = torch.load(model_path)
model.eval()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("/home/zlz/CodeWithDataset/bert-base-uncased")

# 定义预测函数
def predict(title, author, abstract):
    # 拼接输入文本
    text = title + ' ' + author + ' ' + abstract
    # 处理文本
    inputs = tokenizer(text, padding='max_length', max_length=128, return_tensors='pt', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(inputs)
    # 转换为二分类结果
    prediction = (output.squeeze() >= 0.5).int().item()
    if prediction == 1:
        return "该文献是医疗相关的文献"
    else:
        return "该文献不是医疗相关的文献"

# 产品介绍内容
product_intro = """
# 医疗文献分类器

这是一个基于 BERT 模型 的医疗文献分类器，旨在帮助用户快速判断一篇文献是否与医疗相关。
该分类器利用了 BERT 强大的文本编码能力，能够有效提取文献中的语义特征，并给出准确的分类结果。

### 功能：
- **输入**：用户可以输入文献的标题、作者和摘要。
- **输出**：系统会根据输入内容，判断该文献是否属于医疗相关领域。

### 使用方法：
1. 在 `Title` 输入框中输入文献的标题。
2. 在 `Author` 输入框中输入文献的作者。
3. 在 `Abstract` 输入框中输入文献的摘要。
4. 点击 `Submit` 按钮，系统会返回分类结果,判断该文献是否与医疗相关。

### 技术背景:

1.该分类器基于 BERT 模型（Bidirectional Encoder Representations from Transformers），是一种预训练的深度学习模型，能够捕捉文本中的上下文信息，广泛应用于自然语言处理任务。

2.数据集：使用了 DataWhaler 提供的医疗文献数据集，经过大量训练，模型能够有效识别医疗相关的内容。

3.模型架构：BERT 模型用于提取文献的语义特征，后续通过一个简单的全连接层进行二分类（医疗相关或非医疗相关）。

### 开发者：
- 作者：zoulinzhuang
- 联系方式：xzxg001@gmail.com

感谢使用本应用！
"""

# 创建Gradio界面
iface = Interface(
    fn=predict, 
    inputs=[Textbox(label="标题"), Textbox(label="作者"), Textbox(label="摘要")], 
    outputs=Textbox(label="预测"),
    title="医疗文献分类器",
    description="请分别输入标题、作者和摘要，以分类是否为医疗文献。",
    examples = [
    ["Accessible Visual Artworks for Blind and Visually Impaired People", "Quero, Luis Cavazos", "Despite the use of tactile graphics and audio guides, blind and visually impaired people still face challenges to experience and understand visual artworks independently at art exhibitions."],
    ["Antioxidant Status of Rat Liver Mitochondria under Conditions of Moderate Hypothermia of Different Duration", "S I Khizrieva,R A Khalilov,A M Dzhafarova,V R Abdullaev,S I Khizrieva,R A Khalilov,A M Dzhafarova,V R Abdullaev", "For evaluation of the contribution of the antioxidant system of mitochondria into the dynamics of changes in the prooxidant status, the content and activity of some of its components were studied under conditions of moderate hypothermia of varying duration. It was found that short-term hypothermia significantly increased superoxide dismutase activity and decreased the levels of low-molecular-weight antioxidants. Increasing the duration of hypothermia to 1 h led to suppression of activities of superoxide dismutase, glutathione reductase, and glutathione peroxidase and a decrease in glutathione content. Further prolongation of hypothermia (to 3 h) was associated with a significant increase in superoxide dismutase and glutathione peroxidase activities and normalization of the rate of glutathione reductase catalysis; the concentration of glutathione increased significantly."]
],
    article=product_intro  # 使用 article 参数添加产品介绍
)

# 启动Gradio应用
iface.launch(share=True, inline=True, debug=True)