from subprocess import call

models = ['res', 'ssp2']
attacks = ['none', 'fgsm', 'pgd']
N = range(1, 11)

for n in N:
	for model in models:
		for attack in attacks:
			call(f'python mimeticnet.py --model {model} --adv {attack} --N {n}')

