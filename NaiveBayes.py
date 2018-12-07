# -*- coding: utf-8 -*-
import jieba


# import re


class NaiveBayes(object):
    def __init__(self, normFilelen, spamFilelen):
        self.validResult = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.wordDict = {}
        self.stopwords = set()
        self.normFilelen = normFilelen
        self.spamFilelen = spamFilelen
        # 获得停用词表
        with open("./input/stop_words.txt", encoding='utf-8') as f:
            self.stopwords = set(line.rstrip() for line in f.readlines())

    def getStopWords(self):
        for line in open("./input/stop_words.txt", encoding='utf-8'):
            self.stopwords.add(line[:len(line) - 1])

    # 获得词典
    def get_word_list(self, filename, label):
        # 分词结果放入res_list
        email = open(filename, encoding='gb2312', errors='ignore').read()
        email = email[email.index("\n\n")::]  # 去除邮件头部
        res_list = list(jieba.cut(email))
        for i in res_list:
            # i = re.compile(r"[^\u4e00-\u9fa5]").sub("", i)  # 过滤掉非中文字符
            if i.strip() != '' and i is not None and i not in self.stopwords:
                if label in self.wordDict.keys():
                    if i in self.wordDict[label].keys():  # 若列表中的词已在词典中，则加1，否则添加进去
                        self.wordDict[label][i] += 1
                    else:
                        self.wordDict[label].setdefault(i, 1)
                else:
                    self.wordDict.setdefault(label, {})
                    self.wordDict[label].setdefault(i, 1)

    # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    def getTestWords(self, testDict):
        wordProbList = {}
        for word, num in testDict.items():
            if word in self.wordDict['spam'].keys():
                if word in self.wordDict['ham'].keys():
                    # 该文件中包含词个数
                    pw_s = self.wordDict['spam'][word] / self.spamFilelen
                    pw_n = self.wordDict['ham'][word] / self.normFilelen
                    ps_w = pw_s / (pw_s + pw_n)
                    wordProbList.setdefault(word, ps_w)
                else:
                    pw_s = self.wordDict['spam'][word] / self.spamFilelen
                    pw_n = 0.01
                    ps_w = pw_s / (pw_s + pw_n)
                    wordProbList.setdefault(word, ps_w)
            else:
                if word in self.wordDict['ham'].keys():
                    pw_s = 0.01
                    pw_n = self.wordDict['ham'][word] / self.normFilelen
                    ps_w = pw_s / (pw_s + pw_n)
                    wordProbList.setdefault(word, ps_w)
                else:
                    wordProbList.setdefault(word, 0.4)  # 若该词不在脏词词典中，概率设为0.4
        # test = sorted(wordProbList.items(), key=lambda d: d[1], reverse=True)[0:15]# 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
        return wordProbList

    # 计算贝叶斯概率
    def calBayes(self, wordList):
        ps_w = 1
        ps_n = 1
        for word, prob in wordList.items():
            # print(word+"/"+str(prob))
            ps_w *= prob
            ps_n *= (1 - prob)
        p = ps_w / (ps_w + ps_n)
        # print(str(ps_w)+"////"+str(ps_n))
        return p

    # 计算预测结果正确率
    def calMetric(self):
        precision = self.validResult['TP'] / \
                    (self.validResult['TP'] + self.validResult['FP'])
        recall = self.validResult['TP'] / \
                 (self.validResult['TP'] + self.validResult['FN'])
        f1_score = 2 * precision * recall / (precision + recall)
        print('Precision: %2f \t Recall: %2f \t F1_score: %2f' %
              (precision, recall, f1_score))

    def judgemail(self, email):
        testDict = {}
        res_list = list(jieba.cut(email))
        for i in res_list:
            # i = re.compile(r"[^\u4e00-\u9fa5]").sub("", i)  # 过滤掉非中文字符
            if i.strip() != '' and i is not None and i not in self.stopwords:
                if i in testDict.keys():  # 若列表中的词已在词典中，则加1，否则添加进去
                    testDict[i] += 1
                else:
                    testDict.setdefault(i, 1)
        return self.calBayes(self.getTestWords(testDict))
