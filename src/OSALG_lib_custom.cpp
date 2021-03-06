#include <string>
#include <algorithm>
#include <vector>
#include <limits>
#include "OSALG_lib_custom.h"

namespace OSALG_custom {

	struct save2 {
		int p;
		int q;

		save2(int p, int q) : p(p), q(q)
		{}

		save2()
		{}
	} typedef SAVE2_type;//secondary list entries

	struct save1 {
		int m;
		int n;
		SAVE2_type pointer;
	
		save1(int m, int n, SAVE2_type pointer) : m(m), n(n), pointer(pointer)
		{}

		save1(int m, int n, int p, int q) : m(m), n(n), pointer(p, q)
		{}

		save1()
		{}
	} typedef SAVE1_type;//primary list entries

	struct al_inf {
		int d_arr_val = 0;
		std::vector<int> f_arr;
		std::vector<bool> e_arr;
		std::vector<SAVE2_type> p_arr;
	} typedef alignment_info;

	void init_first_row(std::vector<alignment_info> &row, int L, std::vector<int> const &u, std::vector<int> const &v) {
		//setting 0,0
		row[0].f_arr[0] = 0;
		row[0].d_arr_val = 0;
		for (int i = 1; i <= 2 * L; ++i) {
			row[0].f_arr[i] = std::numeric_limits<int>::max();
		}

		row[0].p_arr[0].p = 1;
		row[0].p_arr[0].q = 0;
		row[0].e_arr[0] = true;

		for (int i = 1; i <= 2 * L; ++i) {
			row[0].p_arr[i].p = 0;
			row[0].p_arr[i].q = 0;
		}
		//--------------

		// setting 0, i
		for (int i = 1; i < row.size(); ++i) {
			row[i].f_arr[0] = std::numeric_limits<int>::max();

			int d_val_temp = std::numeric_limits<int>::max();

			for (int j = 1; j <= L; ++j) {
				int k = L + j;

				row[i].f_arr[j] = std::numeric_limits<int>::max();
				row[i].f_arr[k] = std::min(row[i - 1].d_arr_val + v[j - 1], row[i - 1].f_arr[k]) + u[j - 1];

				if (row[i].f_arr[k] < d_val_temp) {
					d_val_temp = row[i].f_arr[k];
				}
			}

			row[i].d_arr_val = d_val_temp;

			for (int k = 0; k <= 2 * L; ++k) {
				row[i].e_arr[k] = (row[i].d_arr_val == row[i].f_arr[k]);
				row[i].p_arr[k].p = (row[i].e_arr[k]) ? 1 : 0;
				row[i].p_arr[k].q = 0;
			}
		}
	}

	void init_first_column_element(std::vector<alignment_info> &current_row, std::vector<alignment_info> &previous_row, int L, std::vector<int> const &u, std::vector<int> const &v) {
		current_row[0].f_arr[0] = std::numeric_limits<int>::max();

		int d_val_temp = std::numeric_limits<int>::max();
		for (int j = 1; j <= L; ++j) {
			int k = L + j;

			current_row[0].f_arr[k] = std::numeric_limits<int>::max();
			current_row[0].f_arr[j] = std::min(previous_row[0].d_arr_val + v[j - 1], previous_row[0].f_arr[j]) + u[j - 1];

			if (current_row[0].f_arr[j] < d_val_temp) {
				d_val_temp = current_row[0].f_arr[j];
			}
		}

		current_row[0].d_arr_val = d_val_temp;

		for (int k = 0; k <= 2 * L; ++k) {
			current_row[0].e_arr[k] = (current_row[0].d_arr_val == current_row[0].f_arr[k]);
			current_row[0].p_arr[k].p = (current_row[0].e_arr[k]) ? 1 : 0;
			current_row[0].p_arr[k].q = 0;
		}
	}

	void adr_function(SAVE2_type &pp, int m, int n, std::vector<alignment_info> &row, std::vector<SAVE1_type> &primary_list, int L) {
		pp.p = primary_list.size() + 1;
		pp.q = 0;

		for (int i = 0; i <= 2 * L; ++i) {
			if (row[n].e_arr[i]) {
				primary_list.emplace_back(m, n, row[n].p_arr[i]);
			}
		}
	}

	void link_function(SAVE2_type &pp, int m, int n, std::vector<alignment_info> &row, std::vector<SAVE2_type> &secondary_list, int i) {
		
		pp.p = row[n].p_arr[0].p;
		pp.q = secondary_list.size() + 1;

		secondary_list.emplace_back(row[n].p_arr[i]);
	}

	int diff(char first, char second, int mismatch_score, int match_score) {
		return (first == second) ? match_score : mismatch_score;
	}

	bool check_for_truth(std::vector<alignment_info> const &row, int n, int L) {
		for (int i = 1; i <= 2 * L; ++i) {
			if (row[n].e_arr[i]) return true;
		}

		return false;
	}

	void allocate_subarray_memory(std::vector<alignment_info> &current_row, std::vector<alignment_info> &previous_row, int seq_len, int L) {
		int resize_size = 2 * L + 1;

		for (int i = 0; i < seq_len + 1; ++i) {
			previous_row[i].e_arr.resize(resize_size);
			previous_row[i].f_arr.resize(resize_size);
			previous_row[i].p_arr.resize(resize_size);

			current_row[i].e_arr.resize(resize_size);
			current_row[i].f_arr.resize(resize_size);
			current_row[i].p_arr.resize(resize_size);
		}
	}

	//(i, j) -> (k, l) -> (m, n)
	void editCIGAR(int i, int j, int k, int l, int m, int n, int current_editing_index, std::vector<std::string> &cigars) {
		int temp;
		
		if (k == m) {
			temp = n - l;

			if (temp != 0) {
				cigars[current_editing_index] = std::to_string(temp) + "I" + cigars[current_editing_index];
			}
		}
		else {
			temp = m - k;
			
			if (temp != 0) {
				cigars[current_editing_index] = std::to_string(temp) + ((l==n) ? "D" : "M") + cigars[current_editing_index];
			}
		}

		if (i == k) {
			temp = l - j;

			if (temp != 0) {
				cigars[current_editing_index] = std::to_string(temp) + "I" + cigars[current_editing_index];
			}
		}
		else {
			temp = k - i;

			if (temp != 0) {
				cigars[current_editing_index] = std::to_string(temp) + ((j == l) ? "D" : "M") + cigars[current_editing_index];
			}
		}
	}

	void process_directional_graph(std::vector<SAVE1_type> const &primary_list, std::vector<SAVE2_type> const &secondary_list,
		std::vector<std::string> &cigars, int current_editing_index, int primary_list_index,
		int next_primary_list_index, bool extended_cigar, bool branched) {;
		
		if (primary_list_index == 0) return;

		if (primary_list[primary_list_index].pointer.q != 0 && !branched) {
			cigars.emplace_back(std::string(cigars[cigars.size() - 1]));
			process_directional_graph(primary_list, secondary_list, cigars, current_editing_index + 1, primary_list_index, secondary_list[primary_list[primary_list_index].pointer.q - 1].p - 1, extended_cigar, true);

			// loop for additional branching
			int temp_q = secondary_list[primary_list[primary_list_index].pointer.q - 1].q;
			int temp_edit_index = current_editing_index + 2;
			while (temp_q != 0) {
				cigars.emplace_back(std::string(cigars[cigars.size() - 1]));
				process_directional_graph(primary_list, secondary_list, cigars, temp_edit_index, primary_list_index, secondary_list[temp_q].p - 1, extended_cigar, true);

				temp_q = secondary_list[temp_q].q;
				temp_edit_index++;
			}
		}

		int k, l;
		if (primary_list[next_primary_list_index].m - primary_list[next_primary_list_index].n > primary_list[primary_list_index].m - primary_list[primary_list_index].n) {
			k = primary_list[primary_list_index].m;
			l = primary_list[primary_list_index].m - primary_list[next_primary_list_index].m + primary_list[next_primary_list_index].n;
		}
		else {
			k = primary_list[primary_list_index].n + primary_list[next_primary_list_index].m - primary_list[next_primary_list_index].n;
			l = primary_list[primary_list_index].n;
		}

		editCIGAR(primary_list[next_primary_list_index].m, primary_list[next_primary_list_index].n, k, l, primary_list[primary_list_index].m, primary_list[primary_list_index].n,
			current_editing_index, cigars);

		process_directional_graph(primary_list, secondary_list, cigars, current_editing_index, next_primary_list_index, primary_list[next_primary_list_index].pointer.p - 1, extended_cigar, false);
	}


	void fill_CIGARS_storage(std::vector<SAVE1_type> const &primary_list, std::vector<SAVE2_type> const &secondary_list, std::vector<std::string> &cigars,
		std::string const &seq1, std::string const &seq2, bool extended_cigar) {

		for (int i = primary_list.size() - 1; i >= 0; --i) {
			if (primary_list[i].m != seq1.length() || primary_list[i].n != seq2.length())
				break;

			cigars.emplace_back(std::string());
			process_directional_graph(primary_list, secondary_list, cigars, cigars.size() - 1, i, primary_list[i].pointer.p - 1, extended_cigar, false);
		}
	}

	int long_gaps_alignment(std::string const &seq1, std::string const &seq2, int L,
						std::vector<int> const &u, std::vector<int> const &v, std::vector<std::string> &cigars,
						int match_score, int mismatch_score, bool extended_cigar) {

		if (v.size() < L || u.size() < L) {
			return -1;
		}

		std::vector<SAVE1_type> primary_list;
		primary_list.emplace_back(0, 0, 0, 0);
		std::vector<SAVE2_type> secondary_list;

		std::vector<alignment_info> previous_row(seq2.length() + 1);
		std::vector<alignment_info> current_row(seq2.length() + 1);


		allocate_subarray_memory(current_row, previous_row, seq2.length(), L)

		init_first_row(previous_row, L, u, v);

		for (unsigned int m = 1; m <= seq1.length(); ++m) {
			init_first_column_element(current_row, previous_row, L, u, v);

			for (unsigned int n = 1; n <= seq2.length(); ++n) {
				for (int i = 1; i <= L; ++i) {
					int j = i + L;

					current_row[n].f_arr[i] = std::min(previous_row[n].d_arr_val + v[i - 1], previous_row[n].f_arr[i]) + u[i - 1];
					current_row[n].f_arr[j] = std::min(current_row[n - 1].d_arr_val + v[i - 1], current_row[n - 1].f_arr[j]) + u[i - 1];

					if (previous_row[n].d_arr_val + v[i - 1] < previous_row[n].f_arr[i]) {
						current_row[n].p_arr[i] = previous_row[n].p_arr[0];
					}
					else if (previous_row[n].d_arr_val + v[i - 1] == previous_row[n].f_arr[i]) {
						link_function(current_row[n].p_arr[i], m - 1, n, previous_row, secondary_list, i);
					}
					else {
						current_row[n].p_arr[i] = previous_row[n].p_arr[i];
					}

					if (current_row[n - 1].d_arr_val + v[i - 1] < current_row[n - 1].f_arr[j]) {
						current_row[n].p_arr[j] = current_row[n - 1].p_arr[0];
					}
					else if (current_row[n - 1].d_arr_val + v[i - 1] == current_row[n - 1].f_arr[j]) {
						link_function(current_row[n].p_arr[j], m, n - 1, current_row, secondary_list, j);
					}
					else {
						current_row[n].p_arr[j] = current_row[n - 1].p_arr[j];
					}
				}

				current_row[n].f_arr[0] = previous_row[n - 1].d_arr_val + diff(seq1[m - 1], seq2[n - 1], match_score, mismatch_score);
				current_row[n].d_arr_val = *std::min_element(current_row[n].f_arr.begin(), current_row[n].f_arr.end());

				for (int i = 0; i <= 2 * L; ++i) {
					current_row[n].e_arr[i] = (current_row[n].d_arr_val == current_row[n].f_arr[i]);
				}

				if (current_row[n].e_arr[0] && check_for_truth(previous_row, n - 1, L)) {
					adr_function(current_row[n].p_arr[0], m - 1, n - 1, previous_row, primary_list, L);
				}
				else {
					current_row[n].p_arr[0] = previous_row[n - 1].p_arr[0];
				}

			}

			previous_row = current_row;
		}

		SAVE2_type p_last;
		adr_function(p_last, seq1.length(), seq2.length(), current_row, primary_list, L);

		fill_CIGARS_storage(primary_list, secondary_list, cigars, seq1, seq2, extended_cigar);

		return 0;
	}
}
