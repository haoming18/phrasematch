import os
import datasets, transformers

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd
import numpy as np

os.environ["WANDB_DISABLED"] = "true"

class CFG:
    input_path = '../input/us-patent-phrase-to-phrase-matching/'
    model_path = ['../input/deberta-v3-5folds/',
                  '../input/bert-for-patent-5fold/', 
                  '../input/deberta-large-v1/',
                 ]
    model_num = 3
    num_fold = 5

titles = pd.read_csv('../input/cpc-codes/titles.csv')

test = pd.read_csv(f"{CFG.input_path}test.csv")
test = test.merge(titles, left_on='context', right_on='code')
test['input'] = test['title']+'[SEP]'+test['anchor']
test = test.drop(columns=["context", "code", "class", "subclass", "group", "main_group", "anchor", "title", "section"])

predictions = []
weights = [0.5, 0.3, 0.2]

for i in range (CFG.model_num):   
    tokenizer = AutoTokenizer.from_pretrained(f'{CFG.model_path[i]}fold0')

    def process_test(unit):
            return {
            **tokenizer( unit['input'], unit['target'])
        }
    
    def process_valid(unit):
        return {
        **tokenizer( unit['input'], unit['target']),
        'label': unit['score']
    }
    
    test_ds = datasets.Dataset.from_pandas(test)
    test_ds = test_ds.map(process_test, remove_columns=['id', 'target', 'input', '__index_level_0__'])

    for fold in range(CFG.num_fold):        
        model = AutoModelForSequenceClassification.from_pretrained(f'{CFG.model_path[i]}fold{fold}', 
                                                                   num_labels=1)
        trainer = Trainer(
                model,
                tokenizer=tokenizer,
            )
        
        outputs = trainer.predict(test_ds)
        prediction = outputs.predictions.reshape(-1) * (weights[i] / 5)
        predictions.append(prediction)
        
predictions = np.sum(predictions, axis=0)
# np.clip(predictions, 0.0, 1.0, out=predictions)

submission = datasets.Dataset.from_dict({
    'id': test['id'],
    'score': predictions,
})

submission.to_csv('pushp.csv', index=False)