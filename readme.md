CrossSpot algorithm code by Meng Jiang

Please keep confidential.

1) crossspot.py
	1.1) set [c_local]: count of injected, foreground block, 1000 as default
	1.2) generate_data(): generate random data and inject the block
		see global variables for data information:
			[MAX_NUM_SEED]: number of seeds in the algorithm
			[k_data]: number of modes
			[vec_n_local]: size vector of block
			[vec_n_global]: size vector of data
			[c_global]: capital C for count of the data
	1.3) load_data(): load from file (data.csv) to
		a) data: entry list + value
		b) item2lineno: [k_data] maps, each map is {item:no. entry in [data] (lineno)}
	1.4) CrossSpot Algorithm
	Output: screen output with
		best accuracy performance (maximum F1 score) with precision, recall and F1
		average F1 score

2) crossspot-less-dense.py
	change [c_local] from 3000 down to 400, from denser block to less dense block:
		generate data
		run CrossSpot algorithm
		Output: in report.csv
			best accuracy performance
			avarage F1 score

