import csv,os,statistics,tempfile
from t36_benchmark_d_only_approximations import benchmark,calculate_bd_rate,data_root,dataset_vec,rel_eb_vec,result_path

new_config_vec=(
    {"name":"cache_8","flags":["-d-only-cache-refill-threshold-experimental","8"]},
    {"name":"candidate_5_cache_9","flags":["-order-candidate-num-experimental","5","-d-only-cache-refill-threshold-experimental","9"]},
    {"name":"candidate_5_cache_8","flags":["-order-candidate-num-experimental","5","-d-only-cache-refill-threshold-experimental","8"]},
)

def main():
    with open(result_path,newline="") as result_file:
        old_result_vec=list(csv.DictReader(result_file))
    new_config_name_set={config["name"] for config in new_config_vec}
    old_result_vec=[result for result in old_result_vec if result["config"] not in new_config_name_set]
    new_result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    for dataset in dataset_vec:
        source_dir=os.path.join(data_root,dataset["folder"])
        with tempfile.TemporaryDirectory(prefix=f"unsz_d_only_combined_{dataset['folder']}_",dir=temporary_root) as test_dir:
            os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
            for config in new_config_vec:
                for rel_eb in rel_eb_vec:
                    result=benchmark(dataset,config,rel_eb,test_dir)
                    new_result_vec.append(result)
                    print(f"{dataset['name']} {config['name']} rel_eb={rel_eb}: preprocessing={result['preprocessing']:.6f} s,sorting={result['sorting']:.6f} s,hits={result['cache_hits']},reduced={result['reduced_ref_hits']},misses={result['cache_misses']},br={result['br']:.6f},psnr={result['psnr']:.6f}",flush=True)
    result_vec=[*old_result_vec,*new_result_vec]
    field_name_vec=list(new_result_vec[0].keys())
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=field_name_vec)
        writer.writeheader()
        writer.writerows(result_vec)
    for dataset in dataset_vec:
        dataset_result_vec=[result for result in result_vec if result["dataset"]==dataset["name"]]
        result_map={}
        for result in dataset_result_vec:
            converted_result={key:(float(value) if key not in ("dataset","config") else value) for key,value in result.items()}
            result_map.setdefault(result["config"],[]).append(converted_result)
        for config_name in ("cache_8","candidate_5_cache_9","candidate_5_cache_8"):
            bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map["exact_10"],result_map[config_name])
            mean_preprocessing=statistics.mean(result["preprocessing"] for result in result_map[config_name])
            mean_sorting=statistics.mean(result["sorting"] for result in result_map[config_name])
            print(f"{dataset['name']} {config_name}: preprocessing={mean_preprocessing:.6f} s,sorting={mean_sorting:.6f} s,BD-rate vs exact_10={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
        for config_name in ("candidate_5_cache_9","candidate_5_cache_8"):
            bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map["candidate_5"],result_map[config_name])
            print(f"{dataset['name']} BD-rate({config_name} vs candidate_5)={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    print(f"结果已追加到{result_path}",flush=True)

if __name__=="__main__":
    main()
