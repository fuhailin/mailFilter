class Classifier(object):
    def __init__(self):
        self.validResult = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.stopwords = []

    # 获得停用词表
    def getStopWords(self):
        with open("./input/stop_words.txt", encoding='utf-8') as f:
            self.stopwords = sorted(f.readlines())
        # for line in open("./input/stop_words.txt", encoding='utf-8'):
        #     self.stopwords.append(line.rstrip())

    # 加载邮件数据的label
    def load_label_files(self, label_file):
        label_dict = {'spam': [], 'ham': []}
        for line in open(label_file).readlines():
            label_list = line.strip().split("..")
            label_dict[label_list[0].strip()].append(label_list[1].strip())
        return label_dict

    # 计算预测结果正确率
    def calMetric(self):
        precision = self.validResult['TP'] / (self.validResult['TP'] + self.validResult['FP'])
        recall = self.validResult['TP'] / (self.validResult['TP'] + self.validResult['FN'])
        f1_score = 2 * precision * recall / (precision + recall)
        print('Precision: %f \t Recall: 5f \t F1_score: %f', precision, recall, f1_score)
