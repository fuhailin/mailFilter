# -*- coding: utf-8 -*-
import socket
from timeit import default_timer as timer

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

from naivebayes import NaiveBayes


def main(flag=True):
    if flag:
        start = timer()
        # 加载邮件数据的label
        label_df = pd.read_csv("./input/trec06c/full/index_bak",
                               sep=' ..', names=['label', 'filename'])

        for key in label_df['label'].unique():
            print(key, len(label_df[label_df['label'] == key]))

        train, valid = train_test_split(label_df, test_size=0.2, random_state=2018)

        normFilelen = train[train['label'] == 'ham'].shape[0]
        spamFilelen = train[train['label'] == 'spam'].shape[0]

        model = NaiveBayes(normFilelen, spamFilelen)
        # model.getStopWords()

        for index, row in tqdm(train.iterrows(), total=train.shape[0]):
            # 将每封邮件出现的词保存在wordsList中
            model.get_word_list('./input/trec06c' +
                                row['filename'], row['label'])
        print('训练集学习完毕，已耗时%2fs' % (timer() - start))

        for index, row in tqdm(valid.iterrows(), total=valid.shape[0]):
            if 'test' in model.wordDict.keys():
                model.wordDict['test'].clear()
            model.get_word_list('./input/trec06c' + row['filename'], 'test')
            wordProbList = model.getTestWords(model.wordDict['test'])
            # 对每封邮件得到的15个词计算贝叶斯概率
            trash_p = model.calBayes(wordProbList)
            if row['label'] == 'spam':
                if trash_p > 0.9:
                    model.validResult['TN'] += 1  # trash
                else:
                    model.validResult['FN'] += 1  # normal
            else:
                if trash_p > 0.9:
                    model.validResult['FP'] += 1  # trash
                else:
                    model.validResult['TP'] += 1  # normal
        model.calMetric()
        print('验证集处理完毕，已耗时%2fs' % (timer() - start))
        pickle.dump(model, open('bayes_model.obj', 'wb'))
    else:
        model = pickle.load(open('bayes_model.obj', 'rb'))
        print("模型加载成功!")
    return model


if __name__ == '__main__':
    model = main(flag=False)  # 是否需要从数据集重新训练模型
    host = ''  # Symbolic name meaning all available interfaces
    port = 8888
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(5)
    while True:
        print("等待连接客户端中...")
        conn, addr = s.accept()
        print('已连接的客户端：', addr)
        msg = ""
        while True:
            data = conn.recv(1024)
            if not len(data):
                break
            msg += data.decode()
        conn.close()
        print('*' * 8 + "收到一下新消息" + '*' * 8 + '\n', msg)
        print('*' * 20)
        P = model.judgemail(msg)
        print("P(spam) = ", P)
        if P > 0.9:
            print("归类于垃圾邮箱！")
        else:
            print("归类于重要邮箱！")
