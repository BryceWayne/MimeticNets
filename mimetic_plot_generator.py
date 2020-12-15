from subprocess import call
import os
import pickle
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd

def get_data(path):
	with open(os.path.join(path, 'data_to_save.pickle'), 'rb') as handle:
		data = pickle.load(handle)
	handle.close()
	return data

def improvement(start, end):
	return round(100*(end-start)/start, 2)

def swap(x):
	x[0], x[-1] = x[-1], x[0]
	return x

cwd = os.getcwd()

PATH = os.path.join(cwd,'trained_networks')
files = os.listdir(PATH)

models = {
	'res':"ResNet",
	'ssp2': 'SSP',
}

adv = {
	'fgsm': "FGSM",
	'pgd': "PGD",
	'none': 'Vanilla'
}

files.sort()
table1 = []
table2 = []
# pprint(files)
# exit()

for file in files:
	terms = file.split('_')
	if terms[0] == 'mimeticNet':
		terms = file.split('_')
		path = os.path.join(PATH, file)
		data = get_data(path)
		df = pd.DataFrame(data).T
		imp = improvement(df['val_acc'].iloc[0], df['best_acc'].iloc[-1])
		entry = {'Model': models[terms[2]],
		         'Adversary': adv[terms[3]],
		         'N': terms[1],
				 'Improvement': imp,
				 'Start': df['val_acc'].iloc[0].round(4),
				 'Best':  df['best_acc'].iloc[-1].round(4),
		}
		table2.append(entry)
		# pprint(entry)
		# plt.figure(figsize=(10,6))
		# x = list(range(1, 51))
		# plt.plot(x, df['val_acc'], color='k', label='Val. Accuracy')
		# plt.plot(x, df['best_acc'], color='r', label='Best Val. Accuracy')
		# plt.xlim(1, 50)
		# plt.ylim(df['val_acc'].min(), 1.006168*min(df['best_acc'].max(), 1))
		# plt.grid(alpha=0.618)
		# plt.title(f"MimeticNet - {file} - {improvement(df['val_acc'].iloc[0], df['best_acc'].iloc[-1])}% Improvement")
		# plt.xlabel('Epoch')
		# plt.ylabel('Accuracy (%)')
		# plt.legend(shadow=True)
		# plt.savefig(os.path.join(PATH, 'pictures', f'{terms[2]}_{terms[3]}_{terms[1]}.png'), bbox_inches='tight', dpi=300)
		# # plt.show()
		# plt.close()
	elif len(terms) == 2:
		continue
		path = os.path.join(PATH, file)
		call(f'python mimetic_attack_test.py --model {terms[0]} --load {os.path.join(PATH, file)}')
		data = get_data(os.path.join(PATH, file))
		# print(cwd)
		# print(PATH)
		# print(path)
		# print(terms)
		entry = {'Model': models[terms[0]],
		         'Adversary': adv[terms[1]],
		         'Natural': data['orig_acc']
		         }
		table1.append(entry)
		pprint(entry)

if len(table1) > 0:
	table1 = pd.DataFrame(table1)
	pprint(table1)
	print()
	table1.to_csv('table1.csv', index=False)
	print(table1.to_latex(index=False))

if len(table2) > 0:
	table2 = pd.DataFrame(table2)
	table2.sort_values(by=['Model', 'Adversary', 'N', 'Improvement'])
	# pprint(table2)
	print()
	table2['N'] = table2['N'].astype(int)
	table2['Start'] = 100*table2['Start']
	table2['Best'] = 100*table2['Best']
	table2.to_csv('table2.csv', index=False)
	# print(table2.to_latex(index=False))
	groups = table2.groupby(['Model', 'Adversary'])
	for name, group in groups:
		print(name)
		print()
		# group.to_latex(buf=f'scatter_{"_".join(name)}.tex', index=False)
		group.sort_values(by=['N'])
		group['N'] = group['N'].astype(int)
		group.to_csv(f'table2_{"_".join(name)}.csv', index=False)
		x = group['N'].to_numpy()
		# x = swap(x)
		plt.figure(figsize=(10,6))
		y2 = group['Start'].to_numpy()
		# y2 = swap(y2)
		plt.plot(x, y2, color='#fc0b01', marker='o', linewidth=0, label='Start')
		y1 = group['Best'].to_numpy()
		# y1 = swap(y1)
		plt.plot(x, y1, color='#0175fc', marker='o', linewidth=0, label='Best')
		for iii in range(len(x)):
			plt.vlines(x[iii], y1[iii], y2[iii], 'k', linewidth=2, linestyle='dashed')
		plt.ylabel('Accuracy (%)')
		plt.xlabel('$N$')
		plt.grid(alpha=0.618)
		plt.legend(shadow=True)
		plt.title(" ".join(name))
		# plt.show()
		plt.savefig(f'scatter_{"_".join(name)}.png', bbox_inches='tight', dpi=300)
