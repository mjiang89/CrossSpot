from copy import *
from math import *
from random import *

MAX_NUM_SEED = 100
k_data = 4
vec_n_local = [3,5,8,10]
vec_n_global = [100,100,100,100]
c_global = 10000
ground_truth = [set(range(0,vec_n_local[k])) for k in range(0,k_data)]

def prob_metric(lower_ns,lower_c,upper_ns,upper_c):
	for k in range(0,k_data):
		if lower_ns[k] == 0: return 0.0
		lower_ns[k] = 1.0*lower_ns[k]
		if upper_ns[k] == 0: return 0.0
		upper_ns[k] = 1.0*upper_ns[k]
	if lower_c == 0: return 0.0
	lower_c = 1.0*lower_c
	if upper_c == 0: return 0.0
	upper_c = 1.0*upper_c
	term1 = lower_c*(log(lower_c)-log(upper_c)-1.0)
	term2 = 1.0
	term3 = 0.0
	for k in range(0,k_data):
		term2 *= lower_ns[k]/upper_ns[k]
		term3 += log(lower_ns[k])-log(upper_ns[k])
	ret = term1+upper_c*term2-lower_c*term3
	return ret

fw = open('result.csv','w')
fw.write('c_local,best_precision,best_recall,best_f1score,average_f1score\n')
for c_local in [3000,2500,2000,1500,1200,1000,950,900,850,800,750,700,650,600,550,500,450,400]:
	best_accuracy = [0.0,0.0,0.0]
	average_f1score = 0.0
	# <begin> Generate ER-Poisson data.
	pair2val = {}
	for i in range(0,c_local):
		pair = ''
		for k in range(0,k_data):
			p = randint(0,vec_n_local[k]-1)
			pair += ','+str(p)
		pair = pair[1:]
		if not pair in pair2val:
			pair2val[pair] = 0
		pair2val[pair] += 1
	for i in range(0,c_global-c_local):
		pair = ''
		for k in range(0,k_data):
			p = randint(vec_n_local[k],vec_n_global[k]-1)
			pair += ','+str(p)
		pair = pair[1:]
		if not pair in pair2val:
			pair2val[pair] = 0
		pair2val[pair] += 1
	data,lineno = [],-1
	item2lineno = [{} for k in range(0,k_data)]
	for pair in pair2val:
		lineno += 1
		entry = [0 for k in range(0,k_data+1)]
		arr = pair.split(',')
		for k in range(0,k_data):
			item = int(arr[k])
			if not item in item2lineno[k]:
				item2lineno[k][item] = set()
			item2lineno[k][item].add(lineno)
			entry[k] = item
		entry[k_data] = pair2val[pair]
		data.append(entry)
	# --- Data ready: data [x0,x1,...x(k-1),val] and item2lineno <end>
	# Generate random seed.
	for seedno in range(0,MAX_NUM_SEED):
		seed = [[set() for k in range(0,k_data)],[set() for k in range(0,k_data)],0,0.0]
		for k in range(0,k_data):
			num_item = randint(1,vec_n_global[k])
			list_item = range(0,vec_n_global[k])
			shuffle(list_item)
			for j in range(0,num_item):
				item = list_item[j]
				seed[0][k].add(item)
		# Item sets ==> Lineno sets ==> Count of block ==> Metric.
		block = copy(seed)
		for k in range(0,k_data):
			for item in block[0][k]:
				if item in item2lineno[k]:
					block[1][k] = block[1][k] | item2lineno[k][item]
		linenoset = block[1][0]
		for k in range(1,k_data):
			linenoset = linenoset & block[1][k]
		block[2] = 0
		for lineno in linenoset:
			block[2] += data[lineno][k_data]
		vec_n_block = [len(block[0][k]) for k in range(0,k_data)]
		c_block = block[2]
		block[3] = prob_metric(copy(vec_n_block),copy(c_block),copy(vec_n_global),copy(c_global))
		# Local Search.
		metric_old = block[3]
		while True:
			list_mode = range(0,k_data)
			shuffle(list_mode)
			for k_adjust in list_mode:
				# Adjust mode [k_adjust].
#				print 'Adjusting mode',k_adjust,'...'
				linenoset = set()
				FIRST_K = True
				for k_fixed in range(0,k_data):
					if k_fixed == k_adjust:
						continue
					if FIRST_K:
						linenoset = copy(block[1][k_fixed])
						FIRST_K = False
					else:
						linenoset = linenoset & copy(block[1][k_fixed])
				item2count = {}
				for lineno in linenoset:
					item = data[lineno][k_adjust]
					count = data[lineno][k_data]
					if not item in item2count:
						item2count[item] = 0
					item2count[item] += count
				vec_n_block = [len(block[0][k]) for k in range(0,k_data)]
				sort_item2count = sorted(item2count.items(),key=lambda x:-x[1])
				num_item = len(sort_item2count)
				if num_item == 0:
					continue
				[item,c_block],n = sort_item2count[0],1
				vec_n_block[k_adjust] = n
				metric_best = prob_metric(copy(vec_n_block),copy(c_block),copy(vec_n_global),copy(c_global))
				itemset = set([item])			
				for i in range(1,num_item):
					[item,count] = sort_item2count[i]
					n += 1
					vec_n_block[k_adjust] = n
					metric_curr = prob_metric(copy(vec_n_block),copy(c_block+count),copy(vec_n_global),copy(c_global))
					if metric_curr <= metric_best:
						break
					metric_best = metric_curr
					c_block += count
					itemset.add(item)
				if metric_best > block[3]:
					block[0][k_adjust] = itemset
					block[1][k_adjust] = set()
					for item in itemset:
						block[1][k_adjust] = block[1][k_adjust] | item2lineno[k_adjust][item]
					block[2] = c_block
					block[3] = metric_best
#				print [len(block[0][k]) for k in range(0,k_data)],block[2],block[3]
			if block[3] == metric_old:
				break
			metric_old = block[3]
		# Evaluation.
		prediction = copy(block)
		for k in range(0,k_data):
			prediction[0][k] = prediction[0][k] & ground_truth[k]
		for k in range(0,k_data):
			prediction[1][k].clear()
			for item in prediction[0][k]:
				if item in item2lineno[k]:
					prediction[1][k] = prediction[1][k] | item2lineno[k][item]
		linenoset = prediction[1][0]
		for k in range(1,k_data):
			linenoset = linenoset & prediction[1][k]
		hits = 0
		for lineno in linenoset:
			hits += data[lineno][k_data]
		precision = 0.0
		if prediction[2] > 0:
			precision = 1.0*hits/prediction[2]
		recall = 1.0*hits/c_local
		f1score = 0.0
		if precision+recall > 0:
			f1score = 2*precision*recall/(precision+recall)
#		print precision,recall,f1score
		if f1score >= best_accuracy[2]:
			best_accuracy = [precision,recall,f1score]
		average_f1score += f1score
	average_f1score /= MAX_NUM_SEED
	s = str(c_local)+','+str(best_accuracy[0])+','+str(best_accuracy[1]) \
		+','+str(best_accuracy[2])+','+str(average_f1score)
	fw.write(s+'\n')
	print s
fw.close()

