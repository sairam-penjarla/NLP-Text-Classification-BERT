import os
import torch
import tarfile
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel

CLASS_LIST = ['INSULT', 'ABUSE', 'PROFANITY', 'OTHER', 'EXPLICIT', 'IMPLICIT']
TRAIN_FILE_1 = 'Shared-Task-2019_Data_germeval2019.training_subtask1_2.txt'
TRAIN_FILE_2 = 'Shared-Task-2019_Data_germeval2019.training_subtask3.txt'
MODEL_NAME = "distilbert-base-german-cased"
MODEL_SAVE_PATH = 'best_model'
NUM_TRAIN_EPOCHS = 4
TEST_SIZE = 0.30


class TextClassificationModel:
    """
    A class to handle text classification model training, evaluation, packing, and unpacking.
    """

    def __init__(self, class_list, model_name, num_train_epochs, test_size):
        self.class_list = class_list
        self.num_train_epochs = num_train_epochs
        self.test_size = test_size
        self.model = None
        self.train_args = {
            "reprocess_input_data": True,
            "fp16": False,
            "num_train_epochs": self.num_train_epochs
        }

    def load_data(self, file1, file2):
        df1 = pd.read_csv(file1, sep='\t', lineterminator='\n', encoding='utf8', names=["tweet", "task1", "task2"])
        df2 = pd.read_csv(file2, sep='\t', lineterminator='\n', encoding='utf8', names=["tweet", "task1", "task2"])
        df = pd.concat([df1, df2])
        df['task2'] = df['task2'].str.replace('\r', "")
        df['pred_class'] = df.apply(lambda x: self.class_list.index(x['task2']), axis=1)
        df = df[['tweet', 'pred_class']]
        return df

    def split_data(self, df):
        train_df, test_df = train_test_split(df, test_size=self.test_size)
        return train_df, test_df

    def train_model(self, train_df):
        self.model.train_model(train_df)

    def evaluate_model(self, test_df):
        def f1_multiclass(labels, preds):
            return f1_score(labels, preds, average='micro')

        result, model_outputs, wrong_predictions = self.model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)
        return result

    def save_model(self, folder_path):
        self.model.model.save_pretrained(folder_path)
        self.model.tokenizer.save_pretrained(folder_path)
        self.model.config.save_pretrained(folder_path)

    def load_model(self, MODEL_NAME, from_pretrained):
        if from_pretrained:
            train_args ={"reprocess_input_data": True,
                "fp16":False,
                "num_train_epochs": 4}
            model = ClassificationModel(
                "bert", from_pretrained,
                num_labels=len(CLASS_LIST),
                args=train_args
            )
        else:
            self.model = ClassificationModel(
                "bert", MODEL_NAME,
                num_labels=len(CLASS_LIST),
                args=self.train_args,
                use_cuda=torch.cuda.is_available()
            )
        self.model = model

    def predict(self, tweet):
        predictions, raw_outputs = self.model.predict([tweet])
        return self.class_list[predictions[0]]


if __name__ == "__main__":
    text_classifier = TextClassificationModel(CLASS_LIST, MODEL_NAME, NUM_TRAIN_EPOCHS, TEST_SIZE)

    # Load and preprocess data
    df = text_classifier.load_data(TRAIN_FILE_1, TRAIN_FILE_2)
    print(df.shape)
    print(df.head())

    # Split data
    train_df, test_df = text_classifier.split_data(df)
    print('train shape: ', train_df.shape)
    print('test shape: ', test_df.shape)

    # Train model
    text_classifier.load_model(MODEL_NAME)
    text_classifier.train_model(train_df)

    # Evaluate model
    results = text_classifier.evaluate_model(test_df)
    print(results)

    # Pack and unpack model
    text_classifier.save_model(MODEL_SAVE_PATH)

    # Load the model again from the saved path for prediction
    model = text_classifier.load_model(from_pretrained=MODEL_SAVE_PATH)

    # Predict new tweets
    test_tweet1 = "Meine Mutter hat mir erzählt, dass mein Vater einen Wahlkreiskandidaten nicht gewählt hat, weil der gegen die Homo-Ehe ist"
    print(text_classifier.predict(test_tweet1))  # OTHER

    test_tweet2 = "Frau #Böttinger meine Meinung dazu ist sie sollten uns mit ihrem Pferdegebiss nicht weiter belästigen #WDR"
    print(text_classifier.predict(test_tweet2))  # ABUSE