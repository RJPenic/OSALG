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
#define TRIANGLE_SIZE 2
#define VECTOR_SIZE 32

#define STOP -1
#define MATCH 0
#define INSERT 1
#define DELETE 2
#define DELETE1 3
#define DELETE2 4

namespace OSALG_vector {

	std::vector<char> ge{ 4, 2 };
	std::vector<char> go{ 1, 13 };

	const std::string mem_safety_add = "++++++++++++++++++++++++++++++++";

	std::unordered_map<int, char> CIGAR_map = {
			{MATCH, 'M'},
			{INSERT, 'I'},
			{DELETE, 'D'}
	};

	int get_last_index(int i, std::string const &reference, std::string const &query) {
		int min = std::min(reference.length() + 1, query.length() + 1);

		if(i < min) {
			return i + 1;
		} else if(i < min + labs(query.length() - reference.length())){
			return min;
		}

		return reference.length() + query.length() + 1 - i;
	}

	char calculate_uv_init(int init_border, int i, char ge1, char go1, char ge2, char go2) {
		if(i + 1 < init_border) {
			return - ge1;
		}

		if(i + 1 == init_border) {
			return (i + 1) * (ge1 - ge2) - (go2 - go1) - ge2;
		}

		return - ge2;
	}

	void init_matrices(char **u, char **v, char **x, char **y, char **xe, char **xe, char ge1, char go1, char ge2, char go2, int init_border, std::string const &reference, std::string const &query) {
		x[0][2] = y[0][0] = - ge1 - go1;
		xe[0][2] = ye[0][0] = - ge1 - go1;

		v[0][2] = u[0][0] = calculate_uv_init(init_border, i, ge1, go1, ge2, go2);

		x[0][1]
	}

	void updatePosition(unsigned int& i, unsigned int& j, int parent, bool first_half_diags, std::string const &reference) {
		if(first_half_diags) {
			switch (parent) {
				case MATCH:
					i -= 2;
					j -= 1;
					break;
				case INSERT:
					i -= 1;
					j -= 1;
					break;
				case DELETE:
					i -= 1;
					break;
			}
		} else {
			switch (parent) {
				case MATCH:
					if(i != reference.length() + 1) {
						j += 1;
					}

					i -= 2;
					break;
				case INSERT:
					i -= 1;
					break;
				case DELETE:
					i -= 1;
					j += 1;
					break;
			}
		}
	}

	void construct_CIGAR(char **u, char **v, char **x, char **xe, char **y, char **ye, int matrix_row_num, std::string &cigar, bool extended_cigar, std::string const &reference, std::string const &query) {
		// TO DO ==> traceback
	}

	void compute_vector(std::string const &reference_safe, std::string const &query_safe, int m, int n,
					__m256i const &match_vec, __m256i const &mismatch_vec, __m256i const &go1_neg_vec, __m256i const &ge1_vec,
					__m256i const &go2_neg_vec, __m256i const &ge2_vec,
					char **u, char **v, char **x, char **xe, char **y, char **ye, int i, int j, int up_j, int left_j) {
		//z[i, j]
		__m256i reference_chars = _mm_loadu_si128((__m256i *)&reference_safe[reference_safe.length() - m + 1 - VECTOR_SIZE]);//already reversed
		__m256i query_chars = _mm_loadu_si128((__m256i *)&query_safe[n - 1]);//load chars
		
		__m256i mask = _mm256_cmpeq_epi8(reference_chars, query_chars);
		__m256i z_vec = _mm256_blendv_epi8 (mismatch_vec, match_vec, mask);
		
		__m256i u_prev_vec = _mm256_loadu_si256((__m256i *)&u[i - 1][left_j]); // u[i, j - 1]
		__m256i v_prev_vec = _mm256_loadu_si256((__m256i *)&v[i - 1][up_j]); // v[i - 1, j]
		__m256i x_prev_vec = _mm256_loadu_si256((__m256i *)&x[i - 1][up_j]); // x[i - 1, j]
		__m256i xe_prev_vec = _mm256_loadu_si256((__m256i *)&xe[i - 1][up_j]); // xe[i - 1, j]
		__m256i y_prev_vec = _mm256_loadu_si256((__m256i *)&y[i - 1][left_j]); // y[i, j - 1]
		__m256i ye_prev_vec = _mm256_loadu_si256((__m256i *)&ye[i - 1][left_j]); // ye[i, j - 1]

		__m256i temp =  _mm256_add_epi8(x_prev_vec, v_prev_vec);
		z_vec = _mm256_max_epi8 (z_vec, temp);

		__m256i temp =  _mm256_add_epi8(xe_prev_vec, v_prev_vec);
		z_vec = _mm256_max_epi8 (z_vec, temp);

		__m256i temp =  _mm256_add_epi8(y_prev_vec, u_prev_vec);
		z_vec = _mm256_max_epi8 (z_vec, temp);

		__m256i temp =  _mm256_add_epi8(ye_prev_vec, u_prev_vec);
		z_vec = _mm256_max_epi8 (z_vec, temp);

		//u[i, j] and v[i, j]
		__m256i u_vec = _mm256_sub_epi8(z_vec, v_prev_vec);
		__m256i v_vec = _mm256_sub_epi8(z_vec, u_prev_vec);

		_mm256_storeu_si256((__m256i *)&u[i][j], u_vec);
		_mm256_storeu_si256((__m256i *)&v[i][j], v_vec);
		
		//x[i, j]
		__m256i x_vec = _mm256_sub_epi8(x_prev_vec, u_vec);
		x_vec = _mm256_max_epi8(x_vec, go1_neg_vec);
		x_vec = _mm256_sub_epi8(x_vec, ge1_vec);

		_mm256_storeu_si256((__m256i *)&x[i][j], x_vec);

		//xe[i, j]
		__m256i xe_vec = _mm256_sub_epi8(xe_prev_vec, u_vec);
		xe_vec = _mm256_max_epi8(xe_vec, go2_neg_vec);
		xe_vec = _mm256_sub_epi8(xe_vec, ge2_vec);

		_mm256_storeu_si256((__m256i *)&xe[i][j], xe_vec);

		//y[i, j]
		__m256i y_vec = _mm256_sub_epi8(y_prev_vec, v_vec);
		y_vec = _mm256_max_epi8(y_vec, go1_neg_vec);
		y_vec = _mm256_sub_epi8(y_vec, ge1_vec);

		_mm256_storeu_si256((__m256i *)&y[i][j], y_vec);

		//ye[i, j]
		__m256i ye_vec = _mm256_sub_epi8(ye_prev_vec, v_vec);
		ye_vec = _mm256_max_epi8(ye_vec, go2_neg_vec);
		ye_vec = _mm256_sub_epi8(ye_vec, ge2_vec);

		_mm256_storeu_si256((__m256i *)&ye[i][j], ye_vec);
	}

	void long_gaps_alignment(std::string const &reference, std::string const &query, std::string &cigar, bool extended_cigar) {
		std::string reference_safe = mem_safety_add + reference;
		std::reverse(reference_safe.begin(), reference_safe.end());//use SIMD for string reversal???
		std::string query_safe = query + mem_safety_add;

		int matrix_row_num = reference.length() + query.length() + 1;

		char **u = new char*[matrix_row_num];
		char **v = new char*[matrix_row_num];
		char **x = new char*[matrix_row_num];
		char **y = new char*[matrix_row_num];
		char **xe = new char*[matrix_row_num];
		char **ye = new char*[matrix_row_num];

		int diagonal_size = ((std::min(reference.length(), query.length())) / VECTOR_SIZE) * VECTOR_SIZE + VECTOR_SIZE + 2;

		//TO DO: SMARTER ALLOCATION
		for(int i = 0; i < matrix_row_num; ++i) {
			u[i] = new char[diagonal_size];
			v[i] = new char[diagonal_size];
			x[i] = new char[diagonal_size];
			y[i] = new char[diagonal_size];
			xe[i] = new char[diagonal_size];
			ye[i] = new char[diagonal_size];
		}

		int init_border = ceil((double)(go[1] - go[0]) / (ge[0] - ge[1]) - 1.0);

		init_matrices(u, v, x, y, xe, ye, ge[0], go[0], ge[1], go[1], init_border, reference, query);

		//ge ==> gap extend
		//go ==> gap open
		__m256i ge1_vec = _mm256_set1_epi8(ge[0]);
		__m256i ge2_vec = _mm256_set1_epi8(ge[1]);
		__m256i go1_neg_vec = _mm256_set1_epi8(-go[0]);
		__m256i go2_neg_vec = _mm256_set1_epi8(-go[1]);
		
		__m256i match_vec = _mm256_set1_epi8(MATCH_SCORE);
		__m256i mismatch_vec = _mm256_set1_epi8(MISMATCH_SCORE);

		for(int i = TRIANGLE_SIZE; i < matrix_row_num; ++i) {
			int last_ind = get_last_index(i, reference, query);

			for(int j = 1; j <= last_ind; j += VECTOR_SIZE) {
				if(i <= reference.length()) {
					compute_vector(reference_safe, query_safe, last_ind - j + ((i > query.length()) ? (i - query.length()) : 0), j - 1,
								match_vec, mismatch_vec,
								go1_neg_vec, ge1_vec, go2_neg_vec, ge2_vec,
								u, v, x, xe, y, ye, i, j, j, j - 1);
				} else {
					compute_vector(reference_safe, query_safe, reference.length() + 1 - j, query.length() - last_ind + j - ((i < query.length()) ? (query.length() - i) : 0),
								match_vec, mismatch_vec,
								go1_neg_vec, ge1_vec, go2_neg_vec, ge2_vec,
								u, v, x, xe, y, ye, i, j, j + 1, j);
				}
			}
			
			//for first row and column
			x[i][last_ind] = y[i][0] = - ge[0] - go[0];
			xe[i][last_ind] = ye[i][0] = - ge[1] - go[1];

			v[i][last_ind] = u[i][0] = calculate_uv_init(init_border, i, ge[0], go[0], ge[1], go[1]);

		}

		construct_CIGAR(u, v, x, xe, y, ye, matrix_row_num, cigar, extended_cigar, reference, query);

		//Delete allocated memory
		for(int i = 0; i < matrix_row_num; ++i) {
			delete[] u[i];
			delete[] v[i];
			delete[] x[i];
			delete[] y[i];
			delete[] xe[i];
			delete[] ye[i];
		}

		delete[] u;
		delete[] v;
		delete[] x;
		delete[] y;
		delete[] xe;
		delete[] ye;
	}
}