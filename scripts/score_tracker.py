import re
import matplotlib.pyplot as plt

def get_move_quant_pairs(cigar):
    pairs_together = re.findall("[0-9]+[=,X,I,D,N]", cigar)
    pairs = []
    for p_t in pairs_together:
        match = re.match(r"([0-9]+)([=,X,I,D,N])", p_t, re.I)
        pairs.append(match.groups())
    
    return pairs

if __name__ == "__main__":
#    cigar = input("Enter CIGAR string: ")
#    
#    match_score = input("Enter match score: ")
#    mismatch_score = input("Enter mismatch penalty(negative): ")
#    gap_open = input("Enter gap open score(negative): ")
#    gap_extend = input("Enter gap extend penalty(negative): ")
    cigar = "34=8D2=22D4=74N1=1D4=66N4=170N3=59N3=2D3=35N1=2D1=2D4=28D"
    match_score = 2
    mismatch_score = -4
    gap_open = -4
    gap_extend = -2
    
    pairs = get_move_quant_pairs(cigar)
    #print(pairs)
    scores = []
    scores.append(0)
    
    for quant, move in pairs:
        if move == '=':
            for i in range(int(quant)):
                scores.append(scores[-1] + match_score)
        elif move == 'X':
            for i in range(int(quant)):
                scores.append(scores[-1] + mismatch_score)
        else:
            scores.append(scores[-1] + gap_open + gap_extend)
            for i in range(int(quant) - 1):
                scores.append(scores[-1] + gap_extend)
            
    plt.plot(scores)
    plt.ylabel('Score')
    plt.grid(True)
    print(- 467051 + 469334)
    plt.axvline(x=- 467051 + 469334)
    plt.show()
            
    