"""Reference: https://medium.com/atlas-research/ner-for-clinical-text-7c73caddd180"""
import config
import glob
import re
import pandas as pd

from sklearn import preprocessing
from tqdm import tqdm

def cal_tag_distribution(concept_filenames):
  # Create dictionary for concept type (tag) distribution
  tag_counter_dict = {}
  for fname in concept_filenames:
    with open(fname, "r") as f:
      text = f.read().strip()
    for row in text.split("\n"):
      tag = row.split('"')[-2]
      if tag_counter_dict.get(tag)!=None:
        tag_counter_dict[tag]+=1
      else:
        tag_counter_dict[tag]=1

  print(tag_counter_dict)


def create_ids(concept_filenames, document_filenames):
  # Create a list of document and concept ids
  concept_ids = []
  document_ids = []

  for fname in concept_filenames:
    fid = re.findall(r'[0-9]+', fname.split("-")[1])[0]
    concept_ids.append(fid)

  for fname in document_filenames:
    fid = re.findall(r'\d+', fname.split("-")[1])[0]
    document_ids.append(fid)
      
  concept_ids = tuple(sorted(concept_ids)) 
  document_ids = tuple(sorted(document_ids))

  intersection = list(set(concept_ids) & set(document_ids))
  if len(intersection) == len(document_ids):
    print("Count of concept files with corresponding doc:", len(intersection))

  print("Total number of files (from one folder):", len(document_ids))

  return concept_ids, document_ids


def create_corpus_lists(file_ids):
  # Create corpus lists (each list of lists for document corpus and concept corpus ordered by document ids)
  document_corpus = []
  concept_corpus = []

  for fid in tqdm(file_ids):
    concept_filepath = f"{config.CONCEPT_FILES_PATH}/clinical-{fid}.txt.con"
    document_filepath = f"{config.DOCUMENT_FILES_PATH}/clinical-{fid}.txt"

    with open(concept_filepath, "r") as f:
      text = f.read().strip().splitlines()
    concept_corpus.append(text)

    with open(document_filepath, "r") as f:
      text = f.read().strip().splitlines()
    document_corpus.append(text)

  return concept_corpus, document_corpus


def create_annotation_df(concept_corpus, concept_ids):
  temp_list = []
  for ind, document in tqdm(enumerate(concept_corpus), total=len(concept_corpus)):
    for row in document:
      text_info = row.split("||")[0].strip()
      type_info = row.split("||")[1].strip()

      text = text_info.split('"')[1].strip()

      offset_start = text_info.split(' ')[-2].strip()
      offset_end = text_info.split(' ')[-1].strip()

      line = offset_start.split(":")[0]

      word_offset_start = int(offset_start.split(":")[1])
      word_offset_end = int(offset_end.split(":")[1])
      length = word_offset_end-word_offset_start + 1

      concept_type = type_info.split('"')[-2]
      
      # Split text into tokens with IOB tags
      first = True  # Set up flag to id start of text
      BIO_tag = "B-"

      if length > 1:  # Isolate text with multiple tokens 
        for offset in range(word_offset_start, word_offset_end+1):
          if first:
            tag_label = BIO_tag + concept_type # Set tag for first word to start with B-
            first = False  # Change flag
          else:
            tag_label = tag_label.replace("B-", "I-")
          temp_list.append([concept_ids[ind], tag_label, line, offset, 1])

      else:
        temp_list.append([concept_ids[ind], BIO_tag + concept_type, line, word_offset_start, length])
        
  cols = ["id", "NER_tag", "row", "offset", "length"]
  df = pd.DataFrame(temp_list, columns=cols)
  df = df.drop(columns=["length"])

  return df

def create_notes_df(document_corpus, concept_ids):
  temp_list = []

  for ind, document in tqdm(enumerate(document_corpus)):

    for row_ind, row in enumerate(document):
      row_split = row.split(" ")
      for word_ind, word in enumerate(row_split):
        word = word.replace("\t", "")
        word_id = concept_ids[ind]
        word_row = row_ind+1  # 1-based indexing 
        word_offset = word_ind # 0-based indexing

        if len(word) > 0 and "|" not in word:
          temp_list.append([word_id, word_row, word_offset, word])

  cols = ["id", "row", "offset", "word"]
  df = pd.DataFrame(temp_list, columns=cols)

  return df

def merge_dfs(annotation_df, notes_df):

  # Check correct data types
  annotation_df[['id', 'row', 'offset']] = annotation_df[['id', 'row', 'offset']].apply(pd.to_numeric)
  annotation_df['NER_tag'] = annotation_df["NER_tag"].astype(str)

  notes_df[['id', 'row', 'offset']] = notes_df[['id', 'row', 'offset']].apply(pd.to_numeric)
  notes_df["word"] = notes_df["word"].astype(str)

  result_df = pd.merge(notes_df, annotation_df, how="left", on=['id', 'row', 'offset'])

  # Check for NaNs (should be only in NER_tag, where NaNs will be replaced with "O" (outside))
  print("Columns with missing data:\n", result_df.isna().any())

  # Replace "O" for "outside" referring to tokens that are not named entities

  result_df = result_df.fillna("O")
  result_df = result_df.drop(columns=["row", "offset"])

  return result_df

def main(concept_filenames, document_filenames):

  # Print tag distribution
  print("Printing tag distribution...")
  cal_tag_distribution(concept_filenames)

  # Create ids
  concept_ids, document_ids = create_ids(concept_filenames, document_filenames)

  # Create corpus lists
  concept_corpus, document_corpus = create_corpus_lists(concept_ids)

  # Create annotation dataframe
  annotation_df = create_annotation_df(concept_corpus, concept_ids)

  # Create notes dataframe
  notes_df = create_notes_df(document_corpus, document_ids)

  result_df = merge_dfs(annotation_df, notes_df)

  return result_df

if __name__ == "__main__":

  # Send lists of document and concept filenames
  main_df = main(
    glob.glob(f"{config.CONCEPT_FILES_PATH}/*.txt.con"),
    glob.glob(f"{config.DOCUMENT_FILES_PATH}/*.txt")
  )

  main_df.to_csv(config.TRAIN_DF, index=False)
