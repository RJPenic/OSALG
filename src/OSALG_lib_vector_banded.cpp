#include <string>
#include <algorithm>
#include <vector>
#include <limits>
#include <unordered_map>
#include <immintrin.h> //AVX
#include <stdlib.h>

#define N_BORDER 30
#define L 2
#define MATCH_SCORE -2
#define MISMATCH_SCORE 4
#define TRIANGLE_SIZE 24
#define BANDWIDTH 24
#define VECTOR_SIZE 8

#define OVERFLOW_CONTROL_VALUE 200

#define STOP -1
#define MATCH 0
#define INSERT 1
#define DELETE 2
#define DELETE1 3
#define DELETE2 4

namespace OSALG_vector {

	std::vector<int> u{ 4, 2 };
	std::vector<int> v{ 1, 13 };

	std::string mem_safety_add = "++++++++++++++++";

	__m256i reverse_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

	std::unordered_map<int, char> CIGAR_map = {
			{MATCH, 'M'},
			{INSERT, 'I'},
			{DELETE, 'D'}
	};

	int diff(char first, char second) {
		return (first == second) ? MATCH_SCORE : MISMATCH_SCORE;
	}

	int get_last_index(int i, std::string const &seq1, std::string const &seq2) {
		int min = std::min(seq1.length() + 1, seq2.length() + 1);

		if(i < min) {
			return i + 1;
		} else if(i < min + labs(seq2.length() - seq1.length())){
			return min;
		}

		return seq1.length() + seq2.length() + 1 - i;
	}

	void process_vector(int **d_mat, int **f1_mat, int **f2_mat, __m256i const &u_1_vec, __m256i const &u_2_vec, __m256i const &v_1_vec, __m256i const &v_2_vec,
				__m256i const &match_vec, __m256i const &mismatch_vec, __m256i const &insert_penalty_vec, int i, int j, int up_j, int left_j, int upper_left_j,
 				int m, int n, int memory_safety_len,
				char const *seq1_safe, char const *seq2_safe) {

		__m256i second_diagonal_d = _mm256_loadu_si256((__m256i *)&d_mat[i - 1][up_j]);
		//First part of convex function
		__m256i second_diagonal_f1 = _mm256_loadu_si256((__m256i *)&f1_mat[i - 1][up_j]);

		__m256i third_diagonal_f1 = _mm256_add_epi32(second_diagonal_d, v_1_vec);
		third_diagonal_f1 = _mm256_min_epi32(third_diagonal_f1, second_diagonal_f1);
		third_diagonal_f1 = _mm256_add_epi32(third_diagonal_f1, u_1_vec);
		__m256i third_diagonal_d = third_diagonal_f1;
		//Second part of convex function
		__m256i second_diagonal_f2 = _mm256_loadu_si256((__m256i *)&f2_mat[i - 1][up_j]);

		__m256i third_diagonal_f2 = _mm256_add_epi32(second_diagonal_d, v_2_vec);
		third_diagonal_f2 = _mm256_min_epi32(third_diagonal_f2, second_diagonal_f2);
		third_diagonal_f2 = _mm256_add_epi32(third_diagonal_f2, u_2_vec);
		third_diagonal_d = _mm256_min_epi32(third_diagonal_f2, third_diagonal_d);

		__m128i seq1_chars = _mm_loadu_si128((__m128i *)&seq1_safe[m - 1 + memory_safety_len]);
		__m128i seq2_chars = _mm_loadu_si128((__m128i *)&seq2_safe[n - 1]);//load chars

		__m256i seq1_chars_ext = _mm256_cvtepu8_epi32 (seq1_chars);
		__m256i seq2_chars_ext = _mm256_cvtepu8_epi32 (seq2_chars);

		seq1_chars_ext = _mm256_permutevar8x32_epi32(seq1_chars_ext, reverse_mask);

		__m256i mask = _mm256_cmpeq_epi32(seq1_chars_ext, seq2_chars_ext);

		__m256i diff_vec = _mm256_blendv_epi8 (mismatch_vec, match_vec, mask);


		__m256i first_diagonal_d = _mm256_loadu_si256((__m256i *)&d_mat[i - 2][upper_left_j]);

		__m256i third_diagonal_f0 = _mm256_add_epi32(first_diagonal_d, diff_vec);

		third_diagonal_d = _mm256_min_epi32(third_diagonal_f0, third_diagonal_d);

		//Insertion
		second_diagonal_d = _mm256_loadu_si256((__m256i *)&d_mat[i - 1][left_j]);

		__m256i third_diagonal_ins = _mm256_add_epi32(second_diagonal_d, insert_penalty_vec);
		third_diagonal_d = _mm256_min_epi32(third_diagonal_ins, third_diagonal_d);

		//storing results in matrices
		_mm256_storeu_si256((__m256i *)&d_mat[i][j], third_diagonal_d);
		_mm256_storeu_si256((__m256i *)&f1_mat[i][j], third_diagonal_f1);
		_mm256_storeu_si256((__m256i *)&f2_mat[i][j], third_diagonal_f2);
	}

	void calculate_diagonal(int **d_mat, int **f1_mat, int **f2_mat, __m256i const &u_1_vec, __m256i const &u_2_vec, __m256i const &v_1_vec, __m256i const &v_2_vec,
				__m256i const &match_vec, __m256i const &mismatch_vec, __m256i const &insert_penalty_vec, int i, int memory_safety_len,
				char const *seq1_safe, char const *seq2_safe, std::string const &seq1, std::string const &seq2, int offset, int len) {

		int last_ind = get_last_index(i, seq1, seq2);

		for(int j = offset; j < offset + len; j += VECTOR_SIZE) {
			if(i <= seq1.length()) {
				process_vector(d_mat, f1_mat, f2_mat, u_1_vec, u_2_vec, v_1_vec, v_2_vec, match_vec, mismatch_vec, insert_penalty_vec,
						i, j, j, j - 1, j - 1,
						last_ind - j + ((i > seq2.length()) ? (i - seq2.length()) : 0), j - 1,
						memory_safety_len, seq1_safe, seq2_safe);
			} else {
				process_vector(d_mat, f1_mat, f2_mat, u_1_vec, u_2_vec, v_1_vec, v_2_vec, match_vec, mismatch_vec, insert_penalty_vec,
						i, j, j + 1, j, (i == seq1.length() + 1) ? j : j + 1,
						seq1.length() + 1 - j, seq2.length() - last_ind + j - ((i < seq2.length()) ? (seq2.length() - i) : 0),
						memory_safety_len, seq1_safe, seq2_safe);
			}
		}
			
		d_mat[i][offset - 1] = f1_mat[i][offset - 1] = f2_mat[i][offset - 1] =
		d_mat[i][offset + len] = f1_mat[i][offset + len] = f2_mat[i][offset + len] = std::numeric_limits<int>::max() - OVERFLOW_CONTROL_VALUE;
	}

	void init_first_triangle(int **d_mat, int **f1_mat, int **f2_mat, __m256i const &u_1_vec, __m256i const &u_2_vec, __m256i const &v_1_vec, __m256i const &v_2_vec,
				__m256i const &match_vec, __m256i const &mismatch_vec, __m256i const &insert_penalty_vec, int memory_safety_len,
				char const *seq1_safe, char const *seq2_safe, std::string const &seq1, std::string const &seq2, int triangle_size) {
		// diagonal 0
		d_mat[0][0] = d_mat[0][2] = std::numeric_limits<int>::max() - OVERFLOW_CONTROL_VALUE;
		d_mat[0][1] = 0;
		
		f1_mat[0][0] = f1_mat[0][1] = f1_mat[0][2] = std::numeric_limits<int>::max() - OVERFLOW_CONTROL_VALUE;
		f2_mat[0][0] = f2_mat[0][1] = f2_mat[0][2] = std::numeric_limits<int>::max() - OVERFLOW_CONTROL_VALUE;

		for(int i = 1; i < triangle_size; ++i) {
			int last_ind = get_last_index(i, seq1, seq2);
			calculate_diagonal(d_mat, f1_mat, f2_mat, u_1_vec, u_2_vec, v_1_vec, v_2_vec, match_vec, mismatch_vec, insert_penalty_vec, i, memory_safety_len,
						seq1_safe, seq2_safe, seq1, seq2, 1, last_ind);
		}
	}

	void init_last_triangle(int **d_mat, int **f1_mat, int **f2_mat, __m256i const &u_1_vec, __m256i const &u_2_vec, __m256i const &v_1_vec, __m256i const &v_2_vec,
				__m256i const &match_vec, __m256i const &mismatch_vec, __m256i const &insert_penalty_vec, int memory_safety_len,
				char const *seq1_safe, char const *seq2_safe, std::string const &seq1, std::string const &seq2, int num_of_diagonals, int triangle_size) {

		for(int i = num_of_diagonals - triangle_size; i < num_of_diagonals; ++i) {
			int last_ind = get_last_index(i, seq1, seq2);
			calculate_diagonal(d_mat, f1_mat, f2_mat, u_1_vec, u_2_vec, v_1_vec, v_2_vec, match_vec, mismatch_vec, insert_penalty_vec, i, memory_safety_len,
						seq1_safe, seq2_safe, seq1, seq2, 1, last_ind);
		}
	}

	void updatePosition(unsigned int& i, unsigned int& j, int parent, bool first_half_diags, std::string const &seq1) {
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
					if(i != seq1.length() + 1) {
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

	void construct_CIGAR(int **d_mat, int **f1_mat, int **f2_mat, int matrix_row_num, std::string &cigar, bool extended_cigar, std::string const &seq1, std::string const &seq2) {
		unsigned int i = matrix_row_num - 1;
		unsigned int j = 1;

		bool firstIdentified = false;
		int counter;
		char lastChar, c;

		bool del_mode1 = false;
		bool del_mode2 = false;
		bool first_half_diags = false;

		int min_seq_len = std::min(seq1.length(), seq2.length());

		while (true) {
			if(i == 0 && j == 1) break;

			if(i <= seq1.length()) first_half_diags = true;

			short parent;
			int last_ind = get_last_index(i, seq1, seq2);		

			if(first_half_diags && j == 1) {
				parent = DELETE;
			} else if(i <= min_seq_len && j == get_last_index(i, seq1, seq2)) {
				parent = INSERT;
			} else {
				if(del_mode2) {
					parent = DELETE;

					if(!(f2_mat[i][j] == ((first_half_diags ? f2_mat[i - 1][j] : f2_mat[i - 1][j + 1]) + u[1]))) {
						del_mode2 = false;
					}
				} else if(del_mode1) {
					parent = DELETE;

					if(d_mat[i][j] == f2_mat[i][j]) {
						del_mode2 = true;
						continue;
					}

					if(! (f1_mat[i][j] == ((first_half_diags ? f1_mat[i - 1][j] : f1_mat[i - 1][j + 1]) + u[0]))) {
						del_mode1 = false;
					}
				} else {
					if(d_mat[i][j] == f2_mat[i][j]) {
						del_mode2 = true;
						continue;
					} else if(d_mat[i][j] == f1_mat[i][j]) {
						del_mode1 = true;
						continue;
					} else if(d_mat[i][j] == (((first_half_diags) ? d_mat[i - 1][j - 1] : d_mat[i - 1][j]) + u[0])) {
						parent = INSERT;
					} else {
						parent = MATCH;
					}
				}
				/*
				if(del_mode) {
					parent = DELETE;
				}
				*/	
			}

			if(extended_cigar) {
				c = CIGAR_map.at(parent);

				if(c == 'M') {
					bool match;

					if(first_half_diags) {
						match = (seq1[last_ind - j + ((i > seq2.length()) ? (i - seq2.length()) : 0) - 1] == seq2[j - 2]);
					} else {
						match = (seq1[seq1.length() - j] == seq2[seq2.length() - 1 - last_ind + j - ((i < seq2.length()) ? (seq2.length() - i) : 0)]);
					}

					if(match) {
						c = '=';
					} else {
						c = 'X';
					}
				}
			} else {
				c = CIGAR_map.at(parent);
			}

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

			updatePosition(i, j, parent, first_half_diags, seq1);
		}

		cigar = std::to_string(counter) + lastChar + cigar;
	}	
	
	int long_gaps_alignment(std::string const &seq1, std::string const &seq2, std::string &cigar, bool extended_cigar) {
		std::string seq1_safe = mem_safety_add + seq1;
		std::string seq2_safe = seq2 + mem_safety_add;

		const char *seq1_arr = seq1_safe.c_str();
		const char *seq2_arr = seq2_safe.c_str();

		int matrix_row_num = seq1.length() + seq2.length() + 1;

		int **d_mat = new int*[matrix_row_num];
		int **f1_mat = new int*[matrix_row_num];
		int **f2_mat = new int*[matrix_row_num];

		int diagonal_size = ((std::min(seq1.length() + 1, seq2.length() + 1) - 1) / VECTOR_SIZE) * VECTOR_SIZE + VECTOR_SIZE + 2;

		//TO DO ALLOCATION
		for(int i = 0; i < matrix_row_num; ++i) {
			d_mat[i] = new int[diagonal_size];
			f1_mat[i] = new int[diagonal_size];
			f2_mat[i] = new int[diagonal_size];
		}

		__m256i insert_penalty_vec = _mm256_set1_epi32(u[0]);

		__m256i u_1_vec = _mm256_set1_epi32(u[0]);
		__m256i u_2_vec = _mm256_set1_epi32(u[1]);
		__m256i v_1_vec = _mm256_set1_epi32(v[0]);
		__m256i v_2_vec = _mm256_set1_epi32(v[1]);

		__m256i match_vec = _mm256_set1_epi32(MATCH_SCORE);
		__m256i mismatch_vec = _mm256_set1_epi32(MISMATCH_SCORE);

		//printf("TEST1\n");

		init_first_triangle(d_mat, f1_mat, f2_mat, u_1_vec, u_2_vec, v_1_vec, v_2_vec, match_vec, mismatch_vec, insert_penalty_vec, mem_safety_add.length(),
				seq1_arr, seq2_arr, seq1, seq2, TRIANGLE_SIZE);

		//printf("TEST2\n");

		int offset = 1;
		for(int i = TRIANGLE_SIZE; i < matrix_row_num - TRIANGLE_SIZE; ++i) {
			calculate_diagonal(d_mat, f1_mat, f2_mat, u_1_vec, u_2_vec, v_1_vec, v_2_vec, match_vec, mismatch_vec, insert_penalty_vec, i, mem_safety_add.length(),
					seq1_arr, seq2_arr, seq1, seq2, offset, BANDWIDTH);

			int last_ind = get_last_index(i, seq1, seq2);

			if(i < seq1.length()) {
				if(d_mat[offset] > d_mat[offset + BANDWIDTH - 1] && offset + BANDWIDTH - 1 < last_ind) {
					++offset;
				}
			} else {
				if((d_mat[offset] < d_mat[offset + BANDWIDTH - 1] && offset >= 1) || offset + BANDWIDTH - 1 == last_ind) {
					--offset;
				}
			}
		}

		//printf("TEST3\n");

		init_last_triangle(d_mat, f1_mat, f2_mat, u_1_vec, u_2_vec, v_1_vec, v_2_vec, match_vec, mismatch_vec, insert_penalty_vec, mem_safety_add.length(),
				seq1_arr, seq2_arr, seq1, seq2, matrix_row_num, TRIANGLE_SIZE);

		//printf("TEST4\n");

		construct_CIGAR(d_mat, f1_mat, f2_mat, matrix_row_num, cigar, extended_cigar, seq1, seq2);

		//printf("TEST5\n");

		//Deleting allocated memory
		for(int i = 0; i < matrix_row_num; ++i) {
			delete[] d_mat[i];
			delete[] f1_mat[i];
			delete[] f2_mat[i];
		}
		delete[] d_mat;
		delete[] f1_mat;
		delete[] f2_mat;

		return 1;
	}
}
