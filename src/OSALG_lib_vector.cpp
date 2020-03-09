#include <string>
#include <algorithm>
#include <vector>
#include <limits>
#include <unordered_map>
#include <cmath>
#include <immintrin.h> //AVX
#include <stdlib.h>

#define N_BORDER 30
#define MATCH_SCORE 2//changed
#define MISMATCH_SCORE -4//changed
#define VECTORIZATION_START_DIAGONAL 2
#define VECTOR_SIZE 32

#define STOP -1
#define MATCH 0
#define INSERT 1
#define DELETE 2

namespace OSALG_vector {

	std::vector<char> ge{ 4, 2 };
	std::vector<char> go{ 2, 13 };

	const std::string mem_safety_add = "++++++++++++++++++++++++++++++++";

	std::unordered_map<int, char> CIGAR_map = {
			{MATCH, 'M'},
			{INSERT, 'I'},
			{DELETE, 'D'}
	};

	int get_diagonal_len(int i, std::string const &reference, std::string const &query) {
		int min = std::min(reference.length() + 1, query.length() + 1);

		if(i < min) {
			return i + 1;
		} else if(i < min + labs(query.length() - reference.length())){
			return min;
		}

		return reference.length() + query.length() + 1 - i;
	}

	void init_matrices(char **u, char **v, char **x, char **y, char ge1, char go1, std::string const &reference, std::string const &query) {
		x[1][1] = y[1][0] = 0;

		v[1][1] = u[1][0] = 0;
	}

	void updatePosition(unsigned int& m, unsigned int& n, int parent) {
		switch (parent) {
			case MATCH:
				m -= 1;
				n -= 1;
				break;
			case INSERT:
				n -= 1;
				break;
			default:
				m -= 1;
				break;
			}
	}

	void construct_CIGAR(char **u, char **v, char **x, char **y, int matrix_row_num, std::string &cigar, bool extended_cigar, std::string const &reference, std::string const &query) {
		unsigned int m = reference.length();
		unsigned int n = query.length();

		bool firstIdentified = false;
		int counter;
		char lastChar, c;

		bool del_mode = false;
		bool ins_mode = false;

		while(true) {
			short parent;

			if(m == 0 && n == 0) break;

			if(n == 0) {
				parent = DELETE;
			} else if(m == 0) {
				parent = INSERT;
			} else {
				if(del_mode) {
					if(x[m + n - 1][n] != go[0]) {
						del_mode = false;
					}
					
					parent = DELETE;
				} else if(ins_mode) {
					if(y[m + n - 1][n - 1] != go[0]) {
						ins_mode = false;
					}

					parent = INSERT;
				} else {
					if(x[m + n - 1][n] == u[m + n][n]) {
						parent = DELETE;

						if(x[m + n - 1][n] == go[0]) {
							del_mode = true;
						}
						
					} else if(y[m + n - 1][n - 1] == v[m + n][n]) {
						parent = INSERT;
						
						if(y[m + n - 1][n - 1] != go[0]) {
							ins_mode = true;
						}
						
					} else {
						parent = MATCH;
					}
				}
			}

			//update CIGAR
			c = CIGAR_map.at(parent);

			if (firstIdentified && c == lastChar) {
				counter++;
			} else {
				if (firstIdentified) {
					if(counter >= 30 && lastChar == 'D') lastChar = 'N';
					cigar = std::to_string(counter) + lastChar + cigar;
				}
				else {
					firstIdentified = true;
				}

				lastChar = c;
				counter = 1;
			}

			updatePosition(m, n, parent);
		}

		cigar = std::to_string(counter) + lastChar + cigar;
	}

	void compute_vector(std::string const &reference_safe, std::string const &query_safe,
					__m256i const &match_vec, __m256i const &mismatch_vec, __m256i const &go1_vec, __m256i const &ge1_vec,
					char **u, char **v, char **x, char **y, int i, int j, int mem_safety_len) {
		int m = i - j;
		int n = j;

		//z[i, j]
		//printf("%d\n", reference_safe.length() - m + 1 - VECTOR_SIZE - mem_safety_len);
		__m256i reference_chars = _mm256_loadu_si256((__m256i *)&reference_safe.c_str()[reference_safe.length() - m + 1]);//already reversed
		__m256i query_chars = _mm256_loadu_si256((__m256i *)&query_safe.c_str()[n - 1]);//load chars
		
		__m256i mask = _mm256_cmpeq_epi8(reference_chars, query_chars);
		__m256i z_vec = _mm256_blendv_epi8 (mismatch_vec, match_vec, mask);
		z_vec = _mm256_add_epi8(z_vec, go1_vec);
		z_vec = _mm256_add_epi8(z_vec, go1_vec);
		z_vec = _mm256_add_epi8(z_vec, ge1_vec);
		z_vec = _mm256_add_epi8(z_vec, ge1_vec);	

		__m256i u_prev_vec = _mm256_loadu_si256((__m256i *)&u[i - 1][j - 1]); // u[i, j - 1]
		__m256i v_prev_vec = _mm256_loadu_si256((__m256i *)&v[i - 1][j]); // v[i - 1, j]
		__m256i x_prev_vec = _mm256_loadu_si256((__m256i *)&x[i - 1][j]); // x[i - 1, j]
		__m256i y_prev_vec = _mm256_loadu_si256((__m256i *)&y[i - 1][j - 1]); // y[i, j - 1]

		__m256i temp =  _mm256_add_epi8(x_prev_vec, v_prev_vec);
		z_vec = _mm256_max_epi8 (z_vec, temp);

		temp =  _mm256_add_epi8(y_prev_vec, u_prev_vec);
		z_vec = _mm256_max_epi8 (z_vec, temp);

		//u[i, j] and v[i, j]
		__m256i u_vec = _mm256_sub_epi8(z_vec, v_prev_vec);
		__m256i v_vec = _mm256_sub_epi8(z_vec, u_prev_vec);

		_mm256_storeu_si256((__m256i *)&u[i][j], u_vec);
		_mm256_storeu_si256((__m256i *)&v[i][j], v_vec);
		
		//x[i, j]
		__m256i x_vec = _mm256_setzero_si256();
		temp = _mm256_sub_epi8(x_prev_vec, u_vec);
		temp = _mm256_add_epi8(temp, go1_vec);
		x_vec = _mm256_max_epi8(x_vec, temp);

		_mm256_storeu_si256((__m256i *)&x[i][j], x_vec);

		//y[i, j]
		__m256i y_vec = _mm256_setzero_si256();
		temp = _mm256_sub_epi8(y_prev_vec, v_vec);
		temp = _mm256_add_epi8(temp, go1_vec);
		y_vec = _mm256_max_epi8(y_vec, temp);

		_mm256_storeu_si256((__m256i *)&y[i][j], y_vec);

	}

	int get_diagonal_start_column(int i, std::string const &reference, std::string const &query) {
		if(i > reference.length()) return i - reference.length();

		return 0;
	}

	void long_gaps_alignment(std::string const &reference, std::string const &query, std::string &cigar, bool extended_cigar) {
		std::string reference_safe = reference;
		std::reverse(reference_safe.begin(), reference_safe.end());//use SIMD for string reversal???
		std::string query_safe = query;

		int matrix_row_num = reference.length() + query.length() + 1;

		char **u = new char*[matrix_row_num];
		char **v = new char*[matrix_row_num];
		char **x = new char*[matrix_row_num];
		char **y = new char*[matrix_row_num];

		int diagonal_size = query.length() + VECTOR_SIZE;

		//TO DO: SMARTER ALLOCATION ==> ???
		for(int i = 0; i < matrix_row_num; ++i) {
			u[i] = new char[diagonal_size];
			v[i] = new char[diagonal_size];
			x[i] = new char[diagonal_size];
			y[i] = new char[diagonal_size];
		}

		init_matrices(u, v, x, y, ge[0], go[0], reference, query);

		//ge ==> gap extend
		//go ==> gap open
		__m256i ge1_vec = _mm256_set1_epi8(ge[0]);
		__m256i go1_vec = _mm256_set1_epi8(go[0]);
		
		__m256i match_vec = _mm256_set1_epi8(MATCH_SCORE);
		__m256i mismatch_vec = _mm256_set1_epi8(MISMATCH_SCORE);


		//iterate through diagonals
		for(int i = VECTORIZATION_START_DIAGONAL; i < matrix_row_num; ++i) {
			int diagonal_len = get_diagonal_len(i, reference, query);
			int start_j = get_diagonal_start_column(i, reference, query);

			for(int j = start_j; j < diagonal_len + start_j; j += VECTOR_SIZE) {
				compute_vector(reference_safe, query_safe, match_vec, mismatch_vec,
							go1_vec, ge1_vec,
							u, v, x, y, i, j, mem_safety_add.length());
			}
			
			//first row and column initialization
			if(i <= reference.length()) {
				y[i][0] = 0;

				u[i][0] = go[0];
			}

			if(i <= query.length()) {
				x[i][i] = 0;

				v[i][i] = go[0];
			}
		}


		construct_CIGAR(u, v, x, y, matrix_row_num, cigar, extended_cigar, reference, query);

		//Delete allocated memory
		for(int i = 0; i < matrix_row_num; ++i) {
			delete[] u[i];
			delete[] v[i];
			delete[] x[i];
			delete[] y[i];
		}

		delete[] u;
		delete[] v;
		delete[] x;
		delete[] y;
	}
}
