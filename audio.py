import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# 读取pkl文件
with open(r'D:\EA\SIMS\Processed\unaligned_39.pkl', 'rb') as f:
    data = pickle.load(f)

# 获取训练集、验证集和测试集的数据
train_data = data['train']
valid_data = data['valid']
test_data = data['test']

# 提取特征数据和标签数据
train_inputs = train_data['audio']
train_labels = train_data['classification_labels']
valid_inputs = valid_data['audio']
valid_labels = valid_data['classification_labels']
test_inputs = test_data['audio']
test_labels = test_data['classification_labels']


# 将标签映射为情绪标签
def map_labels_to_emotion(labels):
    emotions = []
    for label in labels:
        if label < 0:
            emotions.append('Negative')
        elif label == 0:
            emotions.append('Neutral')
        else:
            emotions.append('Positive')
    return emotions


train_labels = map_labels_to_emotion(train_labels)
valid_labels = map_labels_to_emotion(valid_labels)
test_labels = map_labels_to_emotion(test_labels)

# 将情绪标签转换为数字编码
label_to_index = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
train_labels = [label_to_index[label] for label in train_labels]
valid_labels = [label_to_index[label] for label in valid_labels]
test_labels = [label_to_index[label] for label in test_labels]

# 将numpy数组转换为PyTorch张量
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
valid_inputs = torch.tensor(valid_inputs, dtype=torch.float32)
valid_labels = torch.tensor(valid_labels, dtype=torch.long)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)


def save_model(model, optimizer, best_accuracy, model_path='best_model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
    }, model_path)
    print(f'Model saved as {model_path}')


# 定义LSTM模型

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 计算准确率
def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


# 定义模型参数
input_dim = train_inputs.shape[2]  # 特征维度
hidden_dim = 64  # 隐藏单元数量
output_dim = 3  # 输出维度为情绪数量
num_layers = 6  # 隐藏层的数量
sequence_length = 7

# 初始化模型
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
# 检查是否已经有最好的模型存在，如果有则加载
best_accuracy = 0
model_path = 'best_model.pth'
if os.path.isfile(model_path):
    print(f'Loading model from {model_path}')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_accuracy = checkpoint['best_accuracy']
# 模型训练
num_epochs = 20
batch_size = 64
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # 根据你的输入数据，你可能需要调整这里的训练输入方式
    # 例如，可能需要将 train_inputs 转换为适合 LSTM 的三维张量
    # outputs = model(train_inputs.view(-1, sequence_length, input_dim))
    outputs = model(train_inputs)  # 这里需要根据实际情况调整
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    # 在验证集上进行评估
    model.eval()
    with torch.no_grad():
        val_outputs = model(valid_inputs)
        val_loss = criterion(val_outputs, valid_labels)
        val_accuracy = compute_accuracy(val_outputs, valid_labels)

    # 保存最好的模型
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        save_model(model, optimizer, best_accuracy)

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

# 模型预测
model.eval()
with torch.no_grad():
    test_outputs = model(test_inputs)
    test_accuracy = compute_accuracy(test_outputs, test_labels)
    print(f'Test Accuracy: {test_accuracy:.4f}')
