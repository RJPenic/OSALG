#include "DiagonalDynamicMatrix.h"
#include <algorithm>
#include <stdio.h>

GotohMatrixSet::GotohMatrixSet (int seq1_len, int seq2_len, int vector_size, int l) {
	this->vector_size = vector_size;
	this->seq1_len = seq1_len;
	this->seq2_len = seq2_len;
	this->num_of_diagonals = seq1_len + seq2_len + 1;
	this->l = l;
	
	this->mats = new int**[l];
	
	for(int i = 0; i < l; ++i) {
		mats[i] = new int*[num_of_diagonals];
	}

	int diagonal_size = ((std::min(seq1_len + 1, seq2_len + 1) - 1) / vector_size) * vector_size + vector_size + 2;
	
	for(int i = 0; i < num_of_diagonals; ++i) {
		for(int j = 0; j < l; ++j) {
			mats[j][i] = new int[diagonal_size];
		}
	}
}

GotohMatrixSet::~GotohMatrixSet () {
	for(int i = 0; i < l; ++i) {
		for(int j = 0; j < num_of_diagonals; ++j) {
			delete[] mats[i][j];
		}
	}

	for(int i = 0; i < l; ++i) {
		delete[] mats[i];
	}

	delete[] mats;
}

int* GotohMatrixSet::get_pointer(int i, int j, int k) {
	return &mats[k][i + j][(i + j <= seq1_len) ? (j + 1) : (1 - i + seq1_len)];
}

int GotohMatrixSet::get(int i, int j, int k) {
	int *p = DiagonalDynamicMatrix::get_pointer(i, j, k);
	return *p;
}

void GotohMatrixSet::set(int i, int j, int k, int val) {
	int *p = DiagonalDynamicMatrix::get_pointer(i, j, k);
	*p = val;
}
