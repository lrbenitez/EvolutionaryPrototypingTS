# -*- coding: utf-8 -*-

import os
import ctypes


class result(ctypes.Structure):
	_fields_ = [
		('D', ctypes.c_double),
		('w1', ctypes.POINTER(ctypes.c_int)),
		('w2', ctypes.POINTER(ctypes.c_int)),
		('size', ctypes.c_int)]


dtw_lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'dtwf.so'))
dtwf = dtw_lib.dtw
dtwf.restype = result

fastdtwf = dtw_lib.fastdtw
fastdtwf.restype = result

freeptrf = dtw_lib.freeptr
freeptrf.restype = None


def dtw(x, y):
	"""Function that calculates the distance DTW and the alignment between two time series.

	:return: Distance between series.
	:return: alignment between series
	"""
	x_arr = (ctypes.c_double * len(x))(*x)
	y_arr = (ctypes.c_double * len(y))(*y)

	x_len = len(x)
	y_len = len(y)

	resultado = dtwf(x_arr, y_arr, x_len, y_len)
	D = resultado.D
	size = resultado.size

	w1, w2 = [], []
	for i in range(size):
		w1.append(resultado.w1[i])
		w2.append(resultado.w2[i])

	freeptrf(resultado.w1)
	freeptrf(resultado.w2)

	return D, (w1, w2)


def fastdtw(x, y, radius):
	""" Function that calculates the distance fastDTW
	(https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf) and the alignment between two time
	series.

	:param x:
	:param y:
	:param radius: window radiud

	:return: Distance between series.
	:return: alignment between series.
	"""
	x_arr = (ctypes.c_double * len(x))(*x)
	y_arr = (ctypes.c_double * len(y))(*y)

	x_len = len(x)
	y_len = len(y)

	resultado = fastdtwf(x_arr, y_arr, x_len, y_len, radius)
	D = resultado.D
	size = resultado.size

	w1, w2 = [], []
	for i in range(size):
		w1.append(resultado.w1[i])
		w2.append(resultado.w2[i])

	freeptrf(resultado.w1)
	freeptrf(resultado.w2)

	return D, (w1, w2)
