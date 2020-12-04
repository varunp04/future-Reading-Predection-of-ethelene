import numpy as np





def gene(data, lookback, delay, min_ind, max_ind, shuffle=False, batch_size = 128, step = 6 ):
	if max_ind is None:
		max_ind = len(data) - delay -1
	i=min_ind + lookback
	while 1:
		if shuffle:
			rows = np.random.randint(min_ind + lookback, max_ind, size = batch_size)
		else:
			if i+batch_size >=max_ind:
				i=min_ind + lookback
			rows = np.arange(i,min(i+batch_size,max_ind))
			i +=len(rows)
		samples = np.zeros((len(rows),lookback//step,data.shape[-1]))
		targets = np.zeros((len(rows),))
		for j, row in enumerate(rows):
			indices = range(rows[j]-lookback, rows[j],step)
			samples[j] = data[indices]
			targets[j] = data[rows[j] + delay][1]
		yield samples, targets