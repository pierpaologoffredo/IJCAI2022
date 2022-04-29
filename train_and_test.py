import random
from pandas.core.frame import DataFrame
from transformers_political_debate.src.transformers.models.longformer import LongformerPolitics, LongformerTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import os
import torch
import pandas as pd
from statistics import mean
from torch.optim import AdamW

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def train_jointLoss_sentence_span(	train_df,
									train_limit,
									model_name,
									num_labels,
									num_batch=8,
									epochs=3,
									loss_type='joint'):
	
	# ========================================
	#               Training
	# ========================================
								
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if train_limit == None:
		train_limit = len(train_df)

	MAX_LEN_DIALOGUE= 4096
	MAX_LEN_FALLACY = 128
	lr = 5e-5 
	

	tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
	model = LongformerPolitics.from_pretrained('allenai/longformer-base-4096', num_labels = num_labels)
	print('Using LongformerPoliticsSequenceClassificationExtraFeatures')

	print('Learning Rate: ', lr)
		
	# do batches for sentence-level 
	span_train, labels_train, snippets_train, argRel_train, argComp_train = [], [], [], [], []
	for x in batch(train_df.Dialogue[:train_limit], num_batch):
		span_train.append(list(x.values))
	
	for x in batch(train_df.Fallacy[:train_limit], num_batch):
		labels_train.append(x)
	
	for x in batch(train_df.Snippet[:train_limit], num_batch):
		snippets_train.append(x)
	
	for x in batch(train_df.CompLabel[:train_limit], num_batch):
		argComp_train.append(x)
	
	for x in batch(train_df.RelLAbel[:train_limit], num_batch):
		argRel_train.append(x)

	# do batches for span-level 
	model = model.to(device)
	model.gradient_checkpointing_enable()
	model.train()
	optim = AdamW(model.parameters(), lr=lr)

	for epoch in range(epochs):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
		print('Training...')
		for dialogue_batch, label_batch, snippet_batch, argRel_batch, argComp_batch in zip(span_train, labels_train, snippets_train, argRel_train, argComp_train):
			dialogue_batch = [str(x) for x in dialogue_batch] # in case of 'nan' #utterance
			dialogue_inputs = tokenizer(dialogue_batch, return_tensors="pt" ,padding='max_length', truncation=True, max_length=MAX_LEN_DIALOGUE)
			
			snippet_batch = [str(x) for x in snippet_batch] # in case of 'nan' #snippets
			snippet_inputs = tokenizer(snippet_batch, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LEN_FALLACY)

			label_batch = torch.tensor(list(label_batch))
			
			argRel_batch = [str(x) for x in argRel_batch] # in case of 'nan' #argRel
			argRel_inputs = tokenizer(argRel_batch, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LEN_FALLACY)
			
			argComp_batch = [str(x) for x in argComp_batch] # in case of 'nan' #argComp
			argComp_inputs = tokenizer(argComp_batch, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LEN_FALLACY)
		
			if device != 'cpu':
				dialogue_inputs = dialogue_inputs.to(device)
				snippet_inputs = snippet_inputs.to(device)
				
				argRel_inputs = argRel_inputs.to(device)

				argComp_inputs = argComp_inputs.to(device)
				
				label_batch = label_batch.to(device)
			
			optim.zero_grad()

			outputs = model(**dialogue_inputs, labels=label_batch, span=snippet_inputs, argRel=argRel_inputs, argComp=argComp_inputs)

			joint_loss, sen_logits, span_logits = outputs
			
			print('======== Epoch {:} / {:} ======== Training Joint Loss: {:}'.format(epoch + 1, epochs, joint_loss))			
			
			joint_loss.backward()
			
			optim.step()		

	model.save_pretrained('./fine_tuned_model/LongformerPoliticsSequenceClassification_3IA/')


def predict_jointLoss_sentence_span(dialogue_eval, labels_eval, snippets_eval, 
									dict_labels, 
									argComp_eval = None, 
									argRel_eval = None,
									model_name='Longformer',
									loss_type='joint'):
	# ========================================
	#               Validation
	# ========================================
	# After the completion of each training epoch, measure our performance on
	# our validation set.
	# Tracking variables 

	device = "cuda:0" if torch.cuda.is_available() else "cpu"

	MAX_LEN_DIALOGUE= 4096
	MAX_LEN_FALLACY = 128

	num_labels = 6

	prediction_list = []

	tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
	model = LongformerPolitics.from_pretrained('./fine_tuned_model/LongformerPoliticsSequenceClassification_3IA/', num_labels = num_labels)

	# Getting the time
	now = datetime.now()
	runtime = now.strftime("%d/%m/%Y_%H:%M:%S")
	print("Starting PREDICTION at: ", runtime)


	model = model.to(device)
	model.eval()

	# Looping on the test set
	for dialogue, label, snippet, rel, comp in zip(dialogue_eval, labels_eval, snippets_eval, argRel_eval, argComp_eval):
		predicted_count = 0
		
		## Tokenizing dialogue and snippets
		tokenized_dialogue = tokenizer(str(dialogue), return_tensors="pt", padding='max_length', truncation = True, max_length = MAX_LEN_DIALOGUE)
		
		tokenized_snippet = tokenizer(str(snippet), return_tensors="pt", padding='max_length', truncation = True, max_length= MAX_LEN_FALLACY)
		
		if argRel_eval != None:
			tokenized_argRel = tokenizer(str(rel), return_tensors="pt", padding='max_length', truncation = True, max_length= MAX_LEN_FALLACY)
		else:
			tokenized_argRel = None
		
		if argComp_eval != None:
			tokenized_argComp = tokenizer(str(comp), return_tensors="pt", padding='max_length', truncation = True, max_length= MAX_LEN_FALLACY)
		else:
			argComp_eval = None
		
		if device != 'cpu':
			tokenized_dialogue = tokenized_dialogue.to(device)
			tokenized_snippet = tokenized_snippet.to(device)
			tokenized_argRel = tokenized_argRel.to(device)
			tokenized_argComp = tokenized_argComp.to(device)
		
		## Getting the prediction
		_, pred_span, pred_snippet_logits = model(**tokenized_dialogue, span=tokenized_snippet, argRel=tokenized_argRel, argComp=tokenized_argComp)
		pred_snippet_logits = torch.tensor(pred_snippet_logits[0])


		# Performing the Softmax function from logits
		output_softmax = F.softmax(pred_snippet_logits)
		pred_value, pred_idx = output_softmax.max(0)

		# Getting the string of the label predicted
		decoded_label = get_keys_from_value(dict_labels, pred_idx.item())[0]

		single_prediction = [str(dialogue), str(decoded_label), str(rel), str(comp)]

		#print("Single prediction: ", single_prediction)

		prediction_list.append(single_prediction)
		predicted_count += 1
		#print("Added prediction to the list: ", prediction_list)
		#break

	
	if predicted_count == 0:
		decode_label = "404" 
		single_prediction = [str(snippet), str(decoded_label), str(rel), str(comp)]
		print('predicted_count == 0', single_prediction)
		prediction_list.append(single_prediction)
		# break

	#print('prediction_list:', prediction_list)
	# write predicted list to tab-seperated file

	# Converting prediction list to def
	predictions = pd.DataFrame(prediction_list, columns=["Span", "Fallacy", "argRel", "argComp"])

	# Getting the time 
	now = datetime.now()
	runtime = now.strftime("%d-%m-%Y_%H:%M:%S")
	
	# saving_path = "./other code/SemEval20-datasets/_prediction/"+'sameval20_'+model_name+"_CRF_oversampled_"+str(epochs)+"_"+task+".txt"
	saving_path = "./prediction_outputs_csv/"+model_name+'_'+(loss_type if loss_type is not None else '')+'runtime-'+str(runtime)+".csv"

	# Save predictions in csv file
	predictions.to_csv(saving_path, index=False)
	print('DONE writing output:',saving_path)
	return saving_path

def evalFallacies(true, pred):
	macro, micro, weighted, none = [], [], [], []
	precision, recall, fscore, support = precision_recall_fscore_support(true,pred, average='macro')
	macro.extend((precision, recall, fscore, support))
	print('macro:',precision, recall, fscore, support)

	precision, recall, fscore, support = precision_recall_fscore_support(true,pred, average='micro')
	micro.extend((precision, recall, fscore, support))
	print('micro:', precision, recall, fscore, support)

	precision, recall, fscore, support = precision_recall_fscore_support(true,pred, average='weighted')
	weighted.extend((precision, recall, fscore, support))
	print('weighted:', precision, recall, fscore, support)

	precision, recall, fscore, support = precision_recall_fscore_support(true,pred, average=None)
	none.extend((precision.tolist(), recall.tolist(), fscore.tolist(), support.tolist()))
	print('None:', precision, recall, fscore, support)


	print('\nEvaluation\tclassification_report:')
	class_report = classification_report(true,pred, output_dict=True)
	print(classification_report(true,pred))

	metrics = [macro, micro, weighted, none]
	metrics = [convertNoneToNum(x) for x in metrics]

	return metrics, class_report

def convert_labels(num_list, dict_labels):
	# returns a list of string
	# It converts numbers to labels from dict_labels
	return [get_keys_from_value(dict_labels, value)[0] for i, value in enumerate(num_list)]

def convertNoneToNum(l):
	# returns a list where None is replaced with -1
	return [-1 if elem is None else elem for elem in l]

def avg_metrics(metric_list):
	# return a list of "average"
	# the mean is calculated for "column" of the lists in metric_list
	return list(map(mean, zip(*metric_list)))

def avg_cr(cr_list):
	# returns a list of "average" classification_report
	# taking into account the metrics for each labels
	avg_cr = {}
	AdHominem_lst = []
	AppealtoAuthority_lst = []
	AppealtoEmotion_lst = []
	FalseCause_lst = []
	Slipperyslope_lst = []
	Slogans_lst = []
	accuracy_lst, macro_avg_lst, weighted_avg_lst = [], [], []
	for cr_item in cr_list:
		for i in cr_item:
			if i == 'AdHominem':
				AdHominem_lst.append(cr_item[i])
			elif i == 'AppealtoAuthority':
				AppealtoAuthority_lst.append(cr_item[i])
			elif i == 'AppealtoEmotion':
				AppealtoEmotion_lst.append(cr_item[i])
			elif i == 'FalseCause':
				FalseCause_lst.append(cr_item[i])
			elif i == 'Slipperyslope':
				Slipperyslope_lst.append(cr_item[i])
			elif i == 'Slogans':
				Slogans_lst.append(cr_item[i])
			elif i == 'accuracy':
				accuracy_lst.append(cr_item[i])
			elif i == 'macro avg':
				macro_avg_lst.append(cr_item[i])
			elif i == 'weighted avg':
				weighted_avg_lst.append(cr_item[i])
	
	# Dictionay for store the metrics for each classification_report
	# Every key is a field of the "original" classification_report from sklearn
	avg_cr['AdHominem'] = AdHominem_lst
	avg_cr['AppealtoAuthority'] = AppealtoAuthority_lst
	avg_cr['AppealtoEmotion'] = AppealtoEmotion_lst
	avg_cr['FalseCause'] = FalseCause_lst
	avg_cr['Slipperyslope'] = Slipperyslope_lst
	avg_cr['Slogans'] = Slogans_lst
	avg_cr['accuracy'] = accuracy_lst
	avg_cr['macro_avg'] = macro_avg_lst
	avg_cr['weighted_avg'] = weighted_avg_lst

	# Processing the average for each "key"
	for elem in avg_cr:
		if avg_cr[elem]:
			avg_cr[elem] = (dict(pd.DataFrame([x for x in avg_cr[elem]]).mean()) if all(isinstance(x, dict) for x in avg_cr[elem]) else mean(avg_cr[elem]))
			
	return avg_cr


# Getting the time 
now = datetime.now()
runtime = now.strftime("%d/%m/%Y_%H:%M:%S")

print("Starting at: ", runtime)

fallacy_df = pd.read_csv('merged.csv')

# Identifying fallacies to map them into keys
fallacies = fallacy_df['Fallacy'].unique()
keys = [*range(0,len(fallacies)+1, 1)]


# Dictionary to replace fallacy values with numbers
replaced_values = dict(zip(sorted(fallacies), keys))
print(replaced_values)

fallacy_df['Fallacy'].replace(replaced_values, inplace=True)

fallacy_df = fallacy_df.drop(columns=['FileName', 'Date', 'Subcategory'])

rand_int = random.randint(1,999)

# Splitting the dataframe in train and test
#train, test, train_labels, test_labels = train_test_split(fallacy_df, fallacy_df['Fallacy'], test_size=0.2, random_state=rand_int)

test = fallacy_df.sample(1)

train = fallacy_df.drop(test.index)

# ========================================
#               Training
# ========================================
train_jointLoss_sentence_span(
	train_df = train,
	train_limit = None,
	model_name ="Longformer",
	num_labels = 6,
	loss_type='joint'
)
print("TRAINING DONE")

# ========================================
#               Prediction
# ========================================
pred_csv_path = predict_jointLoss_sentence_span(
	dialogue_eval=test['Dialogue'].tolist(),
	labels_eval=test['Fallacy'].tolist(),
	snippets_eval=test['Snippet'].tolist(),
	argRel_eval=test['RelLAbel'].tolist(),
	argComp_eval=test['CompLabel'].tolist(),
	dict_labels=replaced_values,
	model_name='Longformer',
	loss_type='joint'
)
print("PREDICTION DONE")

# ========================================
#               Evaluation
# ========================================
# Getting the predicted labels from the csv file 
pred_df = pd.read_csv(str(pred_csv_path))
pred = pred_df['Fallacy'].tolist()

# Getting the true labels of from the test set and turing them on "string" labels
true = test['Fallacy'].tolist()
true = convert_labels(true, replaced_values)
print("true labels: ", true)


pred_df = pd.read_csv(str(pred_csv_path))
pred = pred_df['Fallacy'].tolist()
# Performing the metric results and classification report
metrics, cr = evalFallacies(true, pred)

print("METRICS: ", metrics)
print("CLASSIFICATION REPORT: ", cr)

# Getting the time
now = datetime.now()
runtime = now.strftime("%d/%m/%Y_%H:%M:%S")
print("RANDOM INT: ", rand_int)
print("Finished at: ", runtime)
