from transformers import pipeline

clf = pipeline(
    task = 'sentiment-analysis', 
    model = 'SkolkovoInstitute/russian_toxicity_classifier')

text = ['Я обожаю обезьян.',
    	'абаюнда!.']

print(clf(text))
