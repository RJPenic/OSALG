
class GotohMatrixSet {
	int num_of_diagonals;
	int vector_size;
	int seq1_len;
	int seq2_len;
	int l;
	
	int ***mats;

	public:
		GotohMatrixSet (int seq1_len, int seq2_len, int vector_size, int l);
		~GotohMatrixSet ();
		
		int* get_pointer(int i, int j, int k);
		int get(int i, int j, int k);
		void set(int i, int j, int k, int val);

	//template?
};
