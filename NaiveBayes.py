import re
import jieba
import os
import collections


class NaiveBayes(object):
    def __init__(self, *args, **kwargs):
        self.regex = re.compile(u'[\u4e00-\u9fff-aA-zZ]+')
        self.wordlist = {'ham': [], 'spam': []}
        self.maildic = {'ham': {}, 'spam': {}}
        self.ratio = {}
        self.normalnum = 0  # 正常邮件和垃圾邮件数目
        self.trashnum = 0  # 初始为史料库中的统计
        # 随着接收邮件的判断，其值还会变动
        self.stopwords = set()
        self.validResult = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        # 获得停用词表

    def getStopWords(self):
        with open("./input/stop_words.txt", encoding='utf-8') as f:
            self.stopwords = set(line.rstrip() for line in f.readlines())

    def splitByjieba(self, label, filename):
        """
        使用用第三方扩展库结巴中文分词进行分词
        """
        email = open(filename, encoding='gb2312', errors='ignore').read()
        email = email[email.index("\n\n")::]
        res = list(set(jieba.cut(email)) - self.stopwords)
        test = []
        for i in res:
            for j in self.regex.findall(i):
                if j is not None and j.strip() != '' and len(j) > 1:
                    test.append(j)
        # res = [i for i in res if i.strip() != '' and i is not None]
        self.wordlist[label].extend(test)
        if filename not in self.maildic[label]:  # 去重并把每封邮件的分词结果存入字典
            self.maildic[label][filename] = res

    def getNTRatio(self, typ):
        """
        分别计算正常(Normal)邮件和垃圾(Trash)邮件中某词在其邮件总数的比例
        typ:['normal', 'trash']
        """
        counter = collections.Counter(self.wordlist[typ])
        dic = collections.defaultdict(list)
        for word in list(counter):
            dic[word].append(counter[word])
        mailcount = len(self.maildic[typ])
        if typ == 'ham':
            self.normalnum = mailcount
        elif typ == 'spam':
            self.trashnum = mailcount
        for key in dic:
            dic[key][0] = dic[key][0] * 1.0 / mailcount
        return dic

    def getRatio(self):
        """
        计算出所有邮件中包含某个词的比例(比如说10封邮件中有5封包含'我们'这个词，
        那么'我们'这个词出现的频率就是50%，这个词来自所有邮件的分词结果)
        """
        dic_normal_ratio = self.getNTRatio('ham')  # 单词在正常邮件中出现的概率
        dic_trash_ratio = self.getNTRatio('spam')  # 单词在垃圾邮件中出现的概率
        dic_ratio = dic_normal_ratio
        for key in dic_trash_ratio:
            if key in dic_ratio:
                dic_ratio[key].append(dic_trash_ratio[key][0])
            else:
                dic_ratio[key].append(0.01)  # 若某单词只出现在正常邮件或垃圾邮件中
                # 那么我们假定它在没出现类型中的概率为0.01
                dic_ratio[key].append(dic_trash_ratio[key][0])
        for key in dic_ratio:
            if len(dic_ratio[key]) == 1:
                dic_ratio[key].append(0.01)
        return dic_ratio

    def splitsingle(self, email):
        """
        分割单个邮件
        返回分词后的单词列表list
        """
        try:
            string = email.decode('gb2312').encode('utf-8')
        except Exception:
            string = email
        res = set(jieba.cut(string))
        # res = []
        # self.search_in_trie(chars, trie, res)
        res = list(res - self.stopwords)
        return res

    def judge(self, email):
        res = self.splitsingle(email)  # res是分词结果，为list
        ratio_of_words = []  # 记录邮件中每个词在垃圾邮件史料库(init.ratio[key][1])中出现的概率
        for word in res:
            if word in self.ratio:
                ratio_of_words.append((word, self.ratio[word][1]))  # 添加(word, ratio)元祖
            else:
                self.ratio[word] = [0.6, 0.4]  # 如果邮件中的词是第一次出现，那么就假定
                # p(s|w)=0.4
            ratio_of_words.append((word, 0.4))
        ratio_of_words = sorted(ratio_of_words, key=lambda x: x[1], reverse=True)[:15]
        P = 1.0
        rest_P = 1.0
        for word in ratio_of_words:
            '''
            try:
                print(word[0].decode('utf-8'), word[1])
            except:
                print(word[0], word[1])
            '''
            P *= word[1]
            rest_P = rest_P * (1.0 - word[1])

        trash_p = P / (P + rest_P)
        # if trash_p > 0.9:
        #     typ = 'trash'
        # else:
        #     typ = 'normal'
        # self.flush(typ, res)
        return trash_p

    # 计算预测结果正确率
    def calMetric(self):
        precision = self.validResult['TP'] / (self.validResult['TP'] + self.validResult['FP'])
        recall = self.validResult['TP'] / (self.validResult['TP'] + self.validResult['FN'])
        f1_score = 2 * precision * recall / (precision + recall)
        print('Precision: %2f \t Recall: %2f \t F1_score: %2f' % (precision, recall, f1_score))
