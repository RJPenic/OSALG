import re
import os

def print_avg_diff_clst_eval(clusters, evals) :
    count = 0
    sum = 0
    
    for name in clusters:
        print(name)
        print(clusters[name])
        print(evals[name])
        for i, cluster in enumerate(clusters[name]):
            count += 2
            sum += abs(cluster[2] - evals[name][i][0]) + abs(cluster[3] - evals[name][i][1])
    
    print(sum / count)
    
def get_ref_ind(refs, pos):
    i = 0
    sum = 0
    
    for ref in refs:
        sum += len(ref)
        
        if pos < sum:
            break
        
        i += 1
    
    return i
    
if __name__ == "__main__":
    #--- Load reference ---
    ref_file_path = input("Reference file >> ")
    reads_file_path = input("Reads file >> ")
    clusters_file_path = input("Clusters file >> ")
    
    eval_dir_path = input("Evaluation files directory >> ")
    
    refs = []
    with open(ref_file_path, 'r') as f:
        for count, line in enumerate(f, start=1):
            if count % 2 == 0:
                refs.append(line)
                
    print("Reference successfully loaded!")
    #--- Load reads ---
    reads = {}
    with open(reads_file_path, 'r') as f:
        while True:
            name = f.readline().rstrip('\n')
            
            if not name:
                break
            
            seq = f.readline().rstrip('\n')
            reads[name[1:]] = seq
            
    print("Reads successfully loaded!")
    
    #--- Read cluster file ---
    clusters = {}
    with open(clusters_file_path, 'r') as f:
        lines = f.readlines()
        start_clusters = True
        
        for line in lines:
            line = line.rstrip('\n')
            if not line:
                continue
            
            if line.startswith('>'):
                temp = line[1:]
                clusters[temp] = []
            else:
                clusters[temp].append([int(s) for s in re.split('\s|-', line)])
            
    print("Clusters successfully loaded!")
            
    #--- Read evaluation files ---
    evals = {}
    
    for filename in os.listdir(eval_dir_path):
        #something_{index}.txt
            
        with open(eval_dir_path + "/" + filename, 'r') as f:
            print("Reading " + filename + "...")
            while True:
                name = f.readline().rstrip('\n')
                if not name:
                    break
                
                graphmap_res = f.readline().rstrip('\n')
                real_res = f.readline().rstrip('\n')
                
                if name in clusters:
                    chr_ind = get_ref_ind(refs, clusters[name][0][3])
                    to_total = 0
                    for i in range(chr_ind):
                        to_total += len(refs[i])
                else:
                    continue
                
                temp = []
                for unit in real_res.split():
                    borders = unit.split('-')
                    temp.append((int(borders[0]) + to_total, to_total + int(borders[1])))
                    
                evals[name] = temp
        
    print("Evaluation files successfully loaded!")
    print("--------- All files successfully loaded! ---------")
    
    #stats calculation
    print_avg_diff_clst_eval(clusters, evals)    
        
        
        