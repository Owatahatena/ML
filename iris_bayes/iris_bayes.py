#coding:utf-8

import sys
from sklearn import datasets
import math
from collections import defaultdict

class N_Bayes():
    def __init__(self):
        self.categories = set()
        self.features = set()
        self.catcount = {}          # catcount[cat] カテゴリの出現回数
        self.denominator = {}       # denominator[cat] P(word|cat)の分母の値
        self.featurecount = {}

    def train(self,namedata, sizedata):
        # 文書集合からカテゴリを抽出して辞書を初期化
        for data in namedata:
            cat = data
            self.categories.add(cat)
        for cat in self.categories:
            self.featurecount[cat] = defaultdict(int)
            self.catcount[cat] = 0
        # 文書集合からカテゴリと単語をカウント
        for cat,feat in zip(namedata,sizedata):
            cat, features = cat, feat[1:]
            self.catcount[cat] += 1
            for feature in features:
                self.features.add(feature)
                self.featurecount[cat][feature] += 1
        # 単語の条件付き確率の分母の値をあらかじめ一括計算しておく（高速化のため）??????
        for cat in self.categories:
            self.denominator[cat] = sum(self.featurecount[cat].values()) + len(self.features)

    def class_ans(self,feature):
        best = None
        max = -sys.maxsize
        for cat in self.catcount.keys():
            p = self.score(feature, cat)
            if p > max:
                max = p
                best = cat
        return best

    def featureProb(self, feature, cat):
        return float(self.featurecount[cat][feature] + 1) / float(self.denominator[cat])

    def score(self,features,cat):
        total = sum(self.catcount.values())
        score = math.log(float(self.catcount[cat]) / float(total))
        score += math.log(self.featureProb(features, cat))
        return score

    def __str__(self):
        total = sum(self.catcount.values())
        return 'iris: %d, features: %d, categories: %d' % (total, len(self.features), len(self.categories))

if __name__ == '__main__':
    nb = N_Bayes()

    data = datasets.load_iris()
    # 各特徴量の名前
    feature_names = data.target
    # データと品種の対応
    datas = data.data
    nb.train(feature_names,datas)

    for i in range(150):
        print(nb.class_ans(datas[i][1]))



    yes_count = 0
    for i in range(150):
        if nb.class_ans(datas[i][1]) == feature_names[i]:
            yes_count += 1
    print('yes_countis ',yes_count)
    # print(len(feature_names))
    # print(len(datas))
    #
    #
    # a = zip(datas,feature_names)
    # for i,j in zip(datas,feature_names):
    #     print (i[2:],j)
    #print(feature_names[0)
    # print(feature_names)
    # for i in datas:
    #     print(i)
    #nb.train(data)
    # print(feature_names[0])
    # print(datas)
