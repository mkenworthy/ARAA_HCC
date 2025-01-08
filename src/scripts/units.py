def rad2arc(x):
	return x * 206265.0

def arc2rad(x):
	return x / 206265.0

def arc2mas(x):
	return 1000.0 * x

def mas2arc(x):
	return x / 1000.0

def mas2rad(x):
	return arc2rad(mas2arc(x))

def rad2mas(x):
	return arc2mas(rad2arc(x))
