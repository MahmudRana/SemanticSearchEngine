import csv
import numpy as np


# This Method returns all the questions from our created FAQ Data Set
def populate_FAQ_data(FAQ_FILE_path):
    question_list = []
    answer_list = []
    with open(FAQ_FILE_path, mode='r', encoding="utf-8") as faq_csv_file:
        csv_reader = csv.DictReader(faq_csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            question_list.append(row["Question"])
            answer_list.append(row["Answer"])
            line_count += 1
        # print(f'Processed {line_count} lines.')
    return question_list, answer_list

def closest_question(input_question, question_list):
    max = -1
    most_probable_question = ""
    # vec_1 = bc.encode(['where to mail my i-20/visa documents?'])
    vec_1 = bc.encode([input_question])
    for i in range (0, len(question_list)):
      # print(i)
      x = [None]*1
      x[0] = (question_list[i])
      # print(x)
      try:
          vec_2 = bc.encode([x[0]])
          score = np.sum(vec_1 * vec_2, axis=1) / np.linalg.norm(vec_2, axis=1)
          if score>max:
            max = score
            most_probable_question = question_list[i]
      except:
          pass
    return most_probable_question, max
#     print(vec1.shape)

from bert_serving.client import BertClient
bc = BertClient()
vec_1 = bc.encode(["where to mail my docs"])

vec_2 = bc.encode(["where to mail my docs"])
print(vec_1)
print("************")
print(vec_2)
score = np.sum(vec_1 * vec_2, axis=1) / np.linalg.norm(vec_2, axis=1)
print(score)
# print(bc.encode(['First do it', 'then do it right', 'then do it better']))

question_list, answer_list = populate_FAQ_data('/home/rana/Thesis/DrQA/EagleBotCode/Data/FAQ_Data/GSU_FAQ_Data.csv')
temp = closest_question('where to send i-20 documents?', question_list)
print(temp)