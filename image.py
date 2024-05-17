import cv2
from deepface import DeepFace
import os


def process_videos_in_folder(root_folder):
    # 打开一个文本文件用于写入结果
    with open('emotion_results.txt', 'w') as result_file:
        # 遍历总文件夹中的所有子文件夹
        for sub_folder in os.listdir(root_folder):
            sub_folder_path = os.path.join(root_folder, sub_folder)
            if os.path.isdir(sub_folder_path):  # 确保它是一个目录
                for filename in os.listdir(sub_folder_path):
                    if filename.lower().endswith('.mp4'):  # 确保处理的是视频文件
                        input_video = os.path.join(sub_folder_path, filename)
                        process_video(input_video, result_file)


def process_video(input_video, result_file):
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_cnt = 0
    emodict = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        for result in results:
            emotion = result['dominant_emotion']
            if emotion not in emodict:
                emodict[emotion] = 0
            emodict[emotion] += 1
        frame_cnt += 1

    cap.release()

    # 计算比例最大的情感
    max_ratio_emotion = max(emodict, key=emodict.get)
    max_ratio = emodict[max_ratio_emotion] / frame_cnt * 100 if frame_cnt > 0 else 0

    # 将结果写入文本文件
    result_file.write(f"{input_video} - Dominant Emotion: {max_ratio_emotion}, Ratio: {max_ratio:.2f}%\n")

    print(f"\nemodict stats for {input_video}:")
    print("total frames: %d" % frame_cnt)
    for emotion in emodict:
        ratio = emodict[emotion] / frame_cnt * 100 if frame_cnt > 0 else 0
        print(f"{emotion:<10} frame num: {emodict[emotion]:<5} ratio: {ratio:.2f}%")


# 将你的总文件夹路径替换为下面的root_folder变量的值
root_folder = "D:/EA/SIMS/Raw"
process_videos_in_folder(root_folder)
