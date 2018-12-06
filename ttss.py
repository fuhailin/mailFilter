# encoding=utf-8
"""
Created on 2018年12月4日
@author: Hailin
"""

from sklearn.model_selection import train_test_split

from spamEmail import spamEmailBayes

# spam类对象
spam = spamEmailBayes()
# 保存词频的词典
spamDict = {}
normDict = {}
testDict = {}
# 保存每封邮件中出现的词
wordsList = []
wordsDict = {}
# 保存预测结果,key为文件名，值为预测类别
testResult = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
# 分别获得正常邮件、垃圾邮件及测试文件名称列表

label_dict = spam.load_label_files(".input/trec06c/full/index_bak")

for key in label_dict.keys():
    print(key, len(label_dict[key]))

norm_train, norm_valid = train_test_split(label_dict['ham'], test_size=0.2)
spam_train, spam_valid = train_test_split(label_dict['spam'], test_size=0.2)

# norm_train, norm_valid = label_dict['ham'][:-400], label_dict['ham'][-400:]
# spam_train, spam_valid = label_dict['spam'][:-400], label_dict['spam'][-400:]

normFileList = norm_train
spamFileList = spam_train
# 获取训练集中正常邮件与垃圾邮件的数量
normFilelen = len(normFileList)
spamFilelen = len(spamFileList)
# 获得停用词表，用于对停用词过滤
stopList = spam.getStopWords()
# 获得正常邮件中的词频
for fileName in normFileList:
    wordsList.clear()
    fn = "./input/trec06c"+fileName
    content = open(fn, encoding='gb2312', errors='ignore').read()
    # print(content)
    content = content[content.index("\n\n") + 2::]  # 去除头信息
    print(content)
    spam.get_word_list(content, wordsList, stopList)
    # 统计每个词在所有邮件中出现的次数
    spam.addToDict(wordsList, wordsDict)
normDict = wordsDict.copy()

# 获得垃圾邮件中的词频
wordsDict.clear()
for fileName in spamFileList:
    wordsList.clear()
    fn = "./input/trec06c"+fileName
    content = open(fn, encoding='gb2312', errors='ignore').read()
    content = content[content.index("\n\n") + 2::]  # 去除头信息
    spam.get_word_list(content, wordsList, stopList)
    spam.addToDict(wordsList, wordsDict)
spamDict = wordsDict.copy()

# 测试ham邮件
for fileName in norm_valid:
    testDict.clear()
    wordsDict.clear()
    wordsList.clear()
    fn = "./input/trec06c"+fileName
    content = open(fn, encoding='gb2312', errors='ignore').read()
    content = content[content.index("\n\n") + 2::]  # 去除头信息
    spam.get_word_list(content, wordsList, stopList)
    spam.addToDict(wordsList, wordsDict)
    testDict = wordsDict.copy()
    # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    wordProbList = spam.getTestWords(
        testDict, spamDict, normDict, normFilelen, spamFilelen)
    # 对每封邮件得到的15个词计算贝叶斯概率
    p = spam.calBayes(wordProbList, spamDict, normDict)
    if(p > 0.9):
        testRes
        testResult.setdefault(fileName, 1)
    else:
        testResult.setdefault(fileName, 0)

# 测试spam邮件
for fileName in spam_valid:
    testDict.clear()
    wordsDict.clear()
    wordsList.clear()
    fn = "./input/trec06c"+fileName
    content = open(fn, encoding='gb2312', errors='ignore').read()
    content = content[content.index("\n\n") + 2::]  # 去除头信息
    spam.get_word_list(content, wordsList, stopList)
    spam.addToDict(wordsList, wordsDict)
    testDict = wordsDict.copy()
    # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    wordProbList = spam.getTestWords(
        testDict, spamDict, normDict, normFilelen, spamFilelen)
    # 对每封邮件得到的15个词计算贝叶斯概率
    p = spam.calBayes(wordProbList, spamDict, normDict)
    if(p > 0.9):
        testResult.setdefault(fileName, 1)
    else:
        testResult.setdefault(fileName, 0)
# 计算分类准确率（测试集中文件名低于1000的为正常邮件）
testAccuracy = spam.calAccuracy(testResult)
for i, ic in testResult.items():
    print(i+"/"+str(ic))
print(testAccuracy)
