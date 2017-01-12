import json
import pandas as pd

from math import log

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = []

    def train(self, train_data):
        self.train = train_data

        #group classes by class name and add class count column to each row
        self.classes = self.train.groupby('class').count().reset_index()
        self.classes = self.classes.drop(['username'], axis = 1)
        self.classes.columns = ['class', 'count']

        #group classes by name and add all the terms in that class in a column
        agg_msg = self.train.groupby('class').agg(lambda x: ''.join(set(x + ' ')))
        agg_msg = agg_msg.drop('username', axis = 1)

        #add agg_msg df to classes df
        self.classes = self.classes.set_index('class').join(agg_msg)

        #calculate a couple features
        self.total_classes = len(self.classes['count'])
        self.total_documents= self.classes['count'].sum()

        self.all_terms = self.get_all_terms(self.classes['message'].sum().split(' '))

        #Calculate the maximum likelihood estimation for each class
        self.classes['mle'] = self.classes['count'].map(lambda x: x/self.total_documents)

        #Compute the probability for each term to appear in each class
        self.classes['terms'] = self.classes['message'].map(lambda x: self.calculate_terms(x, self.all_terms))

    def get_all_terms(self, all_terms):
        terms = set(filter(lambda x: x != '', all_terms))
        return {'term_set' : terms, 'raw_terms': all_terms, 'unique_count' : len(terms)}

    def calculate_terms(self, msg, all_terms):
        split_msg = msg.split(' ')
        total_terms = 0;
        terms = {}

        #initialize term objects
        for term in all_terms['term_set']:
            terms[term] = {'count': 0}

        for t in split_msg:
            if t is not '':
                total_terms += 1
                terms[t]['count'] += 1

        for key, value in terms.items():
            terms[key]['probability'] = (value['count'] + 1)/(total_terms + all_terms['unique_count'])


        return terms

    def test(self, testdf):
        results = testdf
        results['class'] = results.apply(lambda row: self.test_row(row), axis=1)
        return results

    def test_row(self, row):
        result = ""
        result_evidence = 0

        message = row["message"]

        for index, row in self.classes.iterrows():
            evidence = log(row["mle"]) + self.get_terms_evidence(row["terms"], message)
            if(evidence > result_evidence) or (result == ""):
                result_evidence = evidence
                result = row.name

        return result

    def get_terms_evidence(self, class_terms, message):
        split_msg = message.split(' ')
        f = True
        evidence = -1

        for term in split_msg:
            a = class_terms.get(term, {'probability': 1})
            prob =  log(a['probability'])
            if f:
                evidence = prob
                f = False
            else:
                evidence += prob

        return evidence

if __name__ == "__main__":
    traindf = pd.read_json("train.json")
    testdf = pd.read_json("test.json")

    nv = NaiveBayesClassifier()
    nv.train(traindf)

    t = nv.classes['terms'][0]

    print(nv.test(testdf))
