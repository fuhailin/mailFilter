# encoding=utf-8
'''
Created on 2018年12月4日
@author: Hailin
'''
import jieba
import re
import os


class spamEmailBayes:
    # 获得停用词表
    def getStopWords(self):
        stopList = []
        for line in open("./input/stop_words.txt", encoding='utf-8'):
            stopList.append(line.rstrip())
        return stopList
    # 获得词典

    def get_word_list(self, content, wordsList, stopList):
        # 分词结果放入res_list
        res_list = list(jieba.cut(content))
        for i in res_list:
            i = re.compile(r"[^\u4e00-\u9fff]").sub("", i)  # 过滤掉非中文字符
            if i not in stopList and i.strip() != '' and i != None:
                if i not in wordsList:
                    wordsList.append(i)

    # 若列表中的词已在词典中，则加1，否则添加进去
    def addToDict(self, wordsList, wordsDict):
        for item in wordsList:
            if item in wordsDict.keys():
                wordsDict[item] += 1
            else:
                wordsDict.setdefault(item, 1)

    def get_File_List(self, filePath):
        filenames = os.listdir(filePath)
        return filenames

    # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    def getTestWords(self, testDict, spamDict, normDict, normFilelen, spamFilelen):
        wordProbList = {}
        for word, num in testDict.items():
            if word in spamDict.keys() and word in normDict.keys():
                # 该文件中包含词个数
                pw_s = spamDict[word]/spamFilelen
                pw_n = normDict[word]/normFilelen
                ps_w = pw_s/(pw_s+pw_n)
                wordProbList.setdefault(word, ps_w)
            if word in spamDict.keys() and word not in normDict.keys():
                pw_s = spamDict[word]/spamFilelen
                pw_n = 0.01
                ps_w = pw_s/(pw_s+pw_n)
                wordProbList.setdefault(word, ps_w)
            if word not in spamDict.keys() and word in normDict.keys():
                pw_s = 0.01
                pw_n = normDict[word]/normFilelen
                ps_w = pw_s/(pw_s+pw_n)
                wordProbList.setdefault(word, ps_w)
            if word not in spamDict.keys() and word not in normDict.keys():
                # 若该词不在脏词词典中，概率设为0.4
                wordProbList.setdefault(word, 0.4)
        sorted(wordProbList.items(), key=lambda d: d[1], reverse=True)[0:15]
        return (wordProbList)

    # 计算贝叶斯概率
    def calBayes(self, wordList, spamdict, normdict):
        ps_w = 1
        ps_n = 1

        for word, prob in wordList.items():
            print(word+"/"+str(prob))
            ps_w *= (prob)
            ps_n *= (1-prob)
        p = ps_w/(ps_w+ps_n)
        # print(str(ps_w)+"////"+str(ps_n))
        return p

    # 计算预测结果正确率
    def calAccuracy(self, testResult):
        rightCount = 0
        errorCount = 0
        for name, catagory in testResult.items():
            if (int(name) < 1000 and catagory == 0) or(int(name) > 1000 and catagory == 1):
                rightCount += 1
            else:
                errorCount += 1
        return rightCount/(rightCount+errorCount)

    # 加载邮件数据的label
    def load_label_files(self, label_file):
        label_dict = {'spam': [], 'ham': []}
        for line in open(label_file).readlines():
            label_list = line.strip().split("..")
            label_dict[label_list[0].strip()].append(label_list[1].strip())
        return label_dict

    # 判断邮件中的字符是否是中文
    def check_contain_chinese(self, check_str):
        for ch in check_str.decode('utf-8'):
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
