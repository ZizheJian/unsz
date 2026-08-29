import csv,os,re,subprocess,tempfile
import numpy as np
from scipy.spatial import cKDTree

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"build/s1_compress")
result_path=os.path.join(project_root,"hybrid_selector_spatial.csv")
dataset_vec=(
    {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","time_steps":(0,15,30),"adjust_coord":False},
    {"name":"Ocean-layerThickness","folder":"Ocean","variable":"layerThickness","time_steps":(0,15,30),"adjust_coord":False},
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","time_steps":(0,80,161),"adjust_coord":True},
)
block_size_vec=(64,256,1024,4096)
sample_num_vec=(8,16,32)

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def split_blocks(coord,block_size):
    block_vec=[]
    stack=[np.arange(coord.shape[0],dtype=np.uint32)]
    while stack:
        point_ids=stack.pop()
        if point_ids.size<=block_size:
            block_vec.append(point_ids)
            continue
        current_coord=coord[point_ids]
        split_dim=np.argmax(np.ptp(current_coord,axis=0))
        middle=point_ids.size//2
        order=np.argpartition(current_coord[:,split_dim],middle)
        stack.append(point_ids[order[:middle]])
        stack.append(point_ids[order[middle:]])
    return block_vec

def block_accuracy(label,block_vec,sample_num,rng,repeat_num=10):
    accuracy_vec=[]
    for _ in range(repeat_num):
        correct_num=0
        for point_ids in block_vec:
            selected_ids=rng.choice(point_ids,size=min(sample_num,point_ids.size),replace=False)
            selected_predictor=np.mean(label[selected_ids])>=0.5
            correct_num+=np.count_nonzero(label[point_ids]==selected_predictor)
        accuracy_vec.append(correct_num/label.size)
    return np.mean(accuracy_vec),np.std(accuracy_vec)

def analyze_selector(dataset,time_step,coord,selector,rng):
    valid=selector<2
    valid_coord=coord[valid]
    label=selector[valid]
    fitting_ratio=np.mean(label)
    global_majority_accuracy=max(fitting_ratio,1-fitting_ratio)
    query_num=min(11,label.size)
    _,neighbor_ids=cKDTree(valid_coord).query(valid_coord,k=query_num,workers=1)
    neighbor_ids=np.asarray(neighbor_ids).reshape(-1,query_num)[:,1:]
    result_vec=[]
    for neighbor_num in (1,4,10):
        current_num=min(neighbor_num,neighbor_ids.shape[1])
        agreement=np.mean(label[neighbor_ids[:,:current_num]]==label[:,None])
        random_agreement=fitting_ratio**2+(1-fitting_ratio)**2
        adjusted_agreement=(agreement-random_agreement)/(1-random_agreement) if random_agreement<1 else 0
        result_vec.append({"dataset":dataset,"time_step":time_step,"stat":"neighbor","parameter":current_num,"fitting_ratio":fitting_ratio,"global_majority_accuracy":global_majority_accuracy,"value":agreement,"std":0,"adjusted_value":adjusted_agreement})
    for block_size in block_size_vec:
        block_vec=split_blocks(valid_coord,block_size)
        oracle_correct=sum(max(np.count_nonzero(label[point_ids]==0),np.count_nonzero(label[point_ids]==1)) for point_ids in block_vec)
        result_vec.append({"dataset":dataset,"time_step":time_step,"stat":"block_oracle","parameter":block_size,"fitting_ratio":fitting_ratio,"global_majority_accuracy":global_majority_accuracy,"value":oracle_correct/label.size,"std":0,"adjusted_value":0})
        for sample_num in sample_num_vec:
            accuracy,std=block_accuracy(label,block_vec,sample_num,rng)
            result_vec.append({"dataset":dataset,"time_step":time_step,"stat":f"block_sample_{sample_num}","parameter":block_size,"fitting_ratio":fitting_ratio,"global_majority_accuracy":global_majority_accuracy,"value":accuracy,"std":std,"adjusted_value":0})
    return result_vec

def main():
    rng=np.random.default_rng(0)
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    for dataset in dataset_vec:
        source_dir=os.path.join(data_root,dataset["folder"])
        with tempfile.TemporaryDirectory(prefix=f"unsz_hybrid_spatial_{dataset['folder']}_",dir=temporary_root) as test_dir:
            os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
            for file_id,time_step in enumerate(dataset["time_steps"]):
                os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}"),os.path.join(test_dir,f"hybrid.dat.{file_id}"))
            selector_path=os.path.join(test_dir,"selector")
            arguments=["-data-dir",test_dir,"-variable","hybrid","-time-step",f"[0,{len(dataset['time_steps'])-1}]","-M","REL","0.01","-predictor","hybrid_loo","-hybrid-selector-path",selector_path,"-reverse-mean-distance-only-order-experimental","-prediction-structure",os.path.join(test_dir,"prediction_structure.bin")]
            if dataset["adjust_coord"]:
                arguments.append("-adjust-coord")
            output=run_command([compressor,*arguments])
            point_num=int(re.search(r"^point_num=([0-9]+)$",output,re.MULTILINE).group(1))
            coord=np.fromfile(os.path.join(source_dir,"coords.dat"),dtype=np.float32).reshape(point_num,-1).astype(np.float64)
            if dataset["adjust_coord"]:
                transform_matrix=np.array([[1.320727945153478,0,0],[-0.050022502016393564,1.119176596028389,0],[-0.0022639755300411725,0.00041386801901295816,0.02477688868808782]])
                coord=coord@transform_matrix
            selector_vec=[]
            for file_id,time_step in enumerate(dataset["time_steps"]):
                selector=np.fromfile(f"{selector_path}.{file_id}",dtype=np.uint8)
                if selector.size!=point_num:
                    raise RuntimeError("selector点数不正确")
                selector_vec.append(selector)
                current_result=analyze_selector(dataset["name"],time_step,coord,selector,rng)
                result_vec.extend(current_result)
                neighbor_one=next(result for result in current_result if result["stat"]=="neighbor" and result["parameter"]==1)
                block_256=next(result for result in current_result if result["stat"]=="block_sample_16" and result["parameter"]==256)
                print(f"{dataset['name']} time_step={time_step}: fitting_ratio={neighbor_one['fitting_ratio']:.6f},nearest_agreement={neighbor_one['value']:.6f},adjusted={neighbor_one['adjusted_value']:.6f},block256_sample16={block_256['value']:.6f}",flush=True)
            for first_id in range(len(selector_vec)):
                for second_id in range(first_id+1,len(selector_vec)):
                    valid=(selector_vec[first_id]<2)&(selector_vec[second_id]<2)
                    agreement=np.mean(selector_vec[first_id][valid]==selector_vec[second_id][valid])
                    result_vec.append({"dataset":dataset["name"],"time_step":f"{dataset['time_steps'][first_id]}-{dataset['time_steps'][second_id]}","stat":"cross_time_agreement","parameter":0,"fitting_ratio":0,"global_majority_accuracy":0,"value":agreement,"std":0,"adjusted_value":0})
                    print(f"{dataset['name']} cross_time={dataset['time_steps'][first_id]}-{dataset['time_steps'][second_id]}: agreement={agreement:.6f}",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
