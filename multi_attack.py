from subprocess import call
import os

cwd = os.getcwd()

PATH = os.path.join(cwd,'trained_networks')
files = os.listdir(PATH)
attacks = ['fgsm', 'pgd']

keys = {
 'res':'res_none',
 # 'res':'res_fgsm',
 # 'res_pgd',
 # 'ssp2_fgsm',
 'ssp2':'ssp2_none',
 # 'ssp2_pgd'	
}


for file in files:
	terms = file.split('_')
	if terms[0] == 'mimeticNet':
		n, model, id = terms[1], terms[2], terms[3]
		path = os.path.join(PATH, file)
		network = f'./trained_networks/{keys[model]}'
		for attack in attacks:
			call(f'python mimetic_attack_test.py'\
			     f' --model {model} --id {id} '\
			     f'--attack {attack} --load {path}'\
			     f' --N {1} --network {network}')
	else:
		pass
	

# models = ['res', 'ssp2']
# N = range(1, 6)

# for n in N:
# 	for model in models:
# 		for attack in attacks:
# 			call(f'python mimeticnet.py --model {model} --adv {attack} --N {n}')


