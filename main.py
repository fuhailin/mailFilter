from naivebayes import NaiveBayes
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    model = NaiveBayes()
    model.getStopWords()
    # 加载邮件数据的label
    label_df = pd.read_csv("./input/trec06c/full/index_bak", sep=' ..', names=['label', 'filename'])

    for key in label_df['label'].unique():
        print(key, len(label_df[label_df['label'] == key]))

    train, valid = train_test_split(label_df, test_size=0.2, random_state=2018)

    for index, row in tqdm(train.iterrows(), total=train.shape[0]):
        model.splitByjieba(row['label'], './input/trec06c'+row['filename'])
    model.ratio = model.getRatio()

    # pickle.dump(model, open('bayes_model.obj', 'wb'))

    for index, row in tqdm(valid.iterrows(), total=valid.shape[0]):
        email = open('./input/trec06c'+row['filename'], encoding='gb2312', errors='ignore').read()
        email = email[email.index("\n\n")::]
        trash_p = model.judge(email)
        if row['label'] == 'spam':
            if trash_p > 0.9:
                typ = 'trash'
                model.validResult['TN'] += 1  # trash
            else:
                typ = 'normal'
                model.validResult['FN'] += 1
        else:
            if trash_p > 0.9:
                typ = 'trash'
                model.validResult['FP'] += 1
            else:
                typ = 'normal'
                model.validResult['TP'] += 1
    model.calMetric()
