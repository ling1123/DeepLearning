import matplotlib.pyplot as plt

# 打开文件并读取所有行
with open('emotion_results.txt', 'r') as f:
    lines = f.readlines()

# 初始化情绪类型计数字典
emotions_count = {
    "angry": 0,
    "fear": 0,
    "happy": 0,
    "neutral": 0,
    "sad": 0,
    "surprise": 0
}

# 遍历每一行，找到"Dominant Emotion："后面的单词，并统计情绪类型出现次数
for line in lines:
    # 找到"Dominant Emotion："后面的单词
    index = line.find("Dominant Emotion:")
    if index != -1:  # 如果找到了
        # 获取情绪类型
        emotion = line[index + len("Dominant Emotion:"):].strip().split(",")[0]
        # 检查是否是情绪类型
        if emotion in emotions_count:
            # 统计情绪类型出现次数
            emotions_count[emotion] += 1

# 绘制柱状图并添加标注
rects = plt.bar(list(emotions_count.keys()), emotions_count.values())
plt.title("Emotions Distribution")
plt.xlabel("Emotions")
plt.ylabel("Count")
plt.ylim(0, max(emotions_count.values()) + 1)  # 设置y轴范围
for rect in rects:
    height = rect.get_height()
    plt.annotate('{}'.format(height),
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')
plt.show()
