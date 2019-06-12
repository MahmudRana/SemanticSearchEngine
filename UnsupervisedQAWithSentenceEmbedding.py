# This is code for retrieving short answers from Georgia Southern University Website.

import warnings

from EagleBot.SupervisedQAUsingBERT import retreive_from_BERT
from EagleBot.SupervisedQAUsingDrQA import retreive_from_DrQA

warnings.filterwarnings('ignore')
import pickle
import numpy as np
import pandas as pd
import json
from textblob import TextBlob
import nltk
from scipy import spatial
import torch
# import InferSent.encoder.models
import timeit
import json
from elasticsearch import Elasticsearch


GLOVE_840B_300D_PATH = "/home/rana/Software&Data/Data/WordEmbeddings/glove.840B.300d.txt"
INFERSENT_SENTENCE_EMBEDDING_PATH = "/home/rana/Thesis/DrQA/EagleBot/InferSent/encoder/infersent.allnli.pickle"
DEFAULT_CANDIDATE_DOCUMENT_NUMBER = 3
# ELASTICSEARCH_INDEX = "gsu_website_index"
ELASTICSEARCH_INDEX = "gsu_website_index_for_project_demo"
# ELASTICSEARCH_INDEX = "gsu_website_index_for_project_demo_internationals_only"
# nltk.download()


def handle_unsupervised_and_supervised_qa(input_qa):
    start = timeit.default_timer()

    # Part 1: connecting ElasticSearch Connection


    def connect_elasticsearch():
        _es = None
        _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        if _es.ping():
            print('Yay Connected')
        else:
            print('Awww it could not connect!')
        return _es

    es = connect_elasticsearch()

    # Part 2: Getting the question from the user and search for the top 5 probable link.
    # TODO: Need to check if top 3 also serves the purpose
    # q = input("Enter Your Question: ")
    q = input_qa
    start = timeit.default_timer()

    res= es.search(index='gsu',body={
      "query": {
        "match": {
          "text": {
            "query": q,
            "minimum_should_match": "25%"

          }
        }
      }
    })

    # print(res)
    candidate_url_list = []
    candidate_list_size = 0
    # Sometimes we can't find top 5. On those cases we will select whatever we have got from ElasticSearch (i.e. 1,2,3,4)
    # TODO: Need to handle the 0 return case
    if(res.get("hits").get("total") <=DEFAULT_CANDIDATE_DOCUMENT_NUMBER ):
        candidate_list_size = res.get("hits").get("total")
    else:
        candidate_list_size = DEFAULT_CANDIDATE_DOCUMENT_NUMBER

    print("Most probable ", candidate_list_size ,  " document link is below: ")
    for i in range (0, candidate_list_size):
        # print(res)
        candidate_url_list.append((res.get("hits").get("hits")[i].get("_source").get("link")))
        print((res.get("hits").get("hits")[i].get("_source").get("link")))

    # Part 3: Retrieving the JSON files and add them in a single para for answer extraction
    # TODO: This part will be removed in next release. Later we need to save the files in ElasticSearch seperately. (i.e.: A row can be: Link-> Text-> TopicName)
    print("*****************")
    print("Finding the URLs in ElasticSearch \n")
    # TODO: Delete these hardcoded links
    file1 = open('/home/rana/Thesis/DrQA/EagleBotCode/Data/GSUWebsiteData/JSONs/Internationals_2.json', encoding="utf-8")
    file2 = open('/home/rana/Thesis/DrQA/EagleBotCode/Data/GSUWebsiteData/JSONs/Admissions.json', encoding="utf-8")
    json_data1 = json.load(file1)
    json_data2 = json.load(file2)
    json_data = json_data1+json_data2
    paras = []
    for i in range (len(json_data)):
        if(json_data[i].get("link") in candidate_url_list):
            print(json_data[i].get("link"))
            paras.append(json_data[i].get("text"))
    # print(json_data)

    # Part 4: Finding the most probable answer using facebook's Infersent sentence embedding
    # TODO: need to use other paragraph splitting techniques using fastText or some other NLP libraries
    # TODO: Need to play with other embeddings. Like, Gensim's Doc2Vec
    print("********************")
    print("Finding the Answer.............")
    blob = TextBlob(" ".join(paras))
    sentences = [item.raw for item in blob.sentences]

    # infersent = torch.load('C:/Users/mr07520/PycharmProjects/QASystem/SQuAD-master/InferSent/encoder/infersent.allnli.pickle', map_location=lambda storage, loc: storage)
    # infersent.set_glove_path("C:/Users/mr07520/PycharmProjects/QASystem/SQuAD-master/data/glove.840B.300d.txt")
    infersent = torch.load(INFERSENT_SENTENCE_EMBEDDING_PATH, map_location=lambda storage, loc: storage)
    infersent.set_glove_path(GLOVE_840B_300D_PATH)



    infersent.build_vocab(sentences, tokenize=True)

    dict_embeddings = {}
    for i in range(len(sentences)):
        dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)

    question_embedding = infersent.encode([q], tokenize=True)

    cosine_similarity_dict = {}
    for i in range(0, len(dict_embeddings)):
        key = ((list(dict_embeddings.keys())[i]))
        # This below line computes the cosine similarity
        value = (spatial.distance.cosine(dict_embeddings.get((list(dict_embeddings.keys())[i])), question_embedding))
        cosine_similarity_dict[key] = value


    # print(cosine_similarity_dict)
    # print(min(cosine_similarity_dict))
    # print(min(cosine_similarity_dict, key=cosine_similarity_dict.get))
    print("******** Answer Using Unsupervised Approach(Sentence Embedding) ********* ")
    best_unsupervised_answer = (min(cosine_similarity_dict.items(), key=lambda x: x[1]) )
    print(best_unsupervised_answer)

    documents_in_string = str(blob)
    best_supervised_answer = retreive_from_DrQA(documents_in_string, q)


    stop = timeit.default_timer()

    print('Time: ', stop - start)

    return best_unsupervised_answer, best_supervised_answer



# paras = "Be sure to carefully review the regulations so that you do not violate any laws or start unauthorized employment. Unauthorized employment of any kind is a serious violation of F-1 or J-1 status, so it is important to learn what employment is allowed before you consider a potential job. F-1 students do not require a special employment authorization document. J-1 students need work authorization from a Responsible Officer or Alternate Responsible Officer (RO or ARO). F-1 and J-1 students can work a maximum of 20 hours per week on campus. See Georgia Southern’s Student Employment Center for on-campus opportunities. Off campus employment is not allowed under any circumstances, unless participating in Curricular Practical Training (CPT) or Optional Practical Training (OPT), or if a student has a circumstance involving severe economic hardship. Schedule OPT Appointment & I-20 Request : After attending the OPT Workshop, make sure to schedule an appointment to gather your OPT packet and receive your I-20. In instances where USCIS determines that a student faces financial hardship due to unforeseeable factors beyond their control, a student may be authorized to pursue off-campus unemployment.  If a student believes they have a case, they should work closely with their assigned International Student Advisor to submit an application to USCIS for Severe Economic Hardship Employment."

def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if _es.ping():
        print('Yay Connected')
    else:
        print('Awww it could not connect!')
    return _es


def get_most_probable_n_document_link(q, candidate_document_number):
    es = connect_elasticsearch()

    res = es.search(index=ELASTICSEARCH_INDEX, body={
        "query": {
            "match": {
                "text": {
                    "query": q,
                    # TODO: Mention this in paper and play with it
                    "minimum_should_match": "50%"

                }
            }
        }
    })

    # print(res)
    candidate_url_list = []
    candidate_list_size = 0
    # Sometimes we can't find top 5. On those cases we will select whatever we have got from ElasticSearch (i.e. 1,2,3,4)
    # TODO: Need to handle the 0 return case
    if (res.get("hits").get("total") <= candidate_document_number):
        candidate_list_size = res.get("hits").get("total")
    else:
        candidate_list_size = candidate_document_number

    paras = []
    print("Most probable ", candidate_list_size, " document link is below: ")
    for i in range(0, candidate_list_size):
        # print(res)
        candidate_url_list.append((res.get("hits").get("hits")[i].get("_source").get("link")))
        paras.append((res.get("hits").get("hits")[i].get("_source").get("text")))
        print((res.get("hits").get("hits")[i].get("_source").get("link")))
    return candidate_url_list, paras


def get_most_probable_supervised_answer(candidate_url_list, q):
    # Part 3: Retrieving the JSON files and add them in a single para for answer extraction
    # TODO: This part will be removed in next release. Later we need to save the files in ElasticSearch seperately. (i.e.: A row can be: Link-> Text-> TopicName)
    print("*****************")
    print("Finding the URLs in ElasticSearch \n")
    # TODO: Delete these hardcoded links
    file1 = open('/home/rana/Thesis/DrQA/EagleBotCode/Data/GSUWebsiteData/JSONs/Internationals_2.json',
                 encoding="utf-8")
    file2 = open('/home/rana/Thesis/DrQA/EagleBotCode/Data/GSUWebsiteData/JSONs/Admissions.json', encoding="utf-8")
    json_data1 = json.load(file1)
    json_data2 = json.load(file2)
    json_data = json_data1 + json_data2
    paras = []
    for i in range(len(json_data)):
        if (json_data[i].get("link") in candidate_url_list):
            print(json_data[i].get("link"))
            paras.append(json_data[i].get("text"))
    # print(json_data)

    # Part 4: Finding the most probable answer using facebook's Infersent sentence embedding
    # TODO: need to use other paragraph splitting techniques using fastText or some other NLP libraries
    # TODO: Need to play with other embeddings. Like, Gensim's Doc2Vec
    print("********************")
    print("Finding the Answer.............")
    blob = TextBlob(" ".join(paras))
    documents_in_string = str(blob)
    best_supervised_answer = retreive_from_DrQA(documents_in_string, q)
    return best_supervised_answer


def get_most_probable_supervised_answer_from_all_gsu_sites(paras, q):


    # paras = []
    # for i in range(len(json_data)):
    #     if (json_data[i].get("link") in candidate_url_list):
    #         print(json_data[i].get("link"))
    #         paras.append(json_data[i].get("text"))
    # # print(json_data)

    # Part 4: Finding the most probable answer using facebook's Infersent sentence embedding
    # TODO: need to use other paragraph splitting techniques using fastText or some other NLP libraries
    # TODO: Need to play with other embeddings. Like, Gensim's Doc2Vec
    print("********************")
    print("Finding the Answer.............")
    blob = TextBlob(" ".join(paras))
    documents_in_string = str(blob)
    best_supervised_answer = retreive_from_DrQA(documents_in_string, q)
    return best_supervised_answer


def get_most_probable_answer_using_bert(paras, q):
    print("********************")
    print("Finding the Answer.............")
    blob = TextBlob(" ".join(paras))
    documents_in_string = str(blob)
    most_probable_answer_from_bert = retreive_from_BERT(documents_in_string, q)
    return most_probable_answer_from_bert


def handle_qa_extraction_using_machine_learning(q):

    candidate_url_list, paras = get_most_probable_n_document_link(q, DEFAULT_CANDIDATE_DOCUMENT_NUMBER)
    # most_probable_supervised_answer = get_most_probable_supervised_answer_from_all_gsu_sites(paras, q)
    most_probable_supervised_answer = "test"
    most_probable_answer_from_bert = get_most_probable_answer_using_bert(paras, q)
    # most_probable_answer_from_bert=[]
    # most_probable_answer_from_bert.append("test1")
    # most_probable_answer_from_bert.append("test2")
    return  candidate_url_list, most_probable_supervised_answer, most_probable_answer_from_bert


if __name__ == "__main__":
    while(True):
        print("Input: ")
        q = input()
        # handle_unsupervised_and_supervised_qa(q)
        handle_qa_extraction_using_machine_learning(q)
