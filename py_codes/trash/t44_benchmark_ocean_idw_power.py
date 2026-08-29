import csv,math,os,re,statistics,subprocess,tempfile
import numpy as np

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"ocean_idw_power_benchmark.csv")
summary_path=os.path.join(project_root,"ocean_idw_power_summary.csv")
field_vec=("layerThickness","salinity","temperature","velocityMeridional","velocityX","velocityY","velocityZ","velocityZonal","zTop")
power_vec=(1.0,2.0,3.0,4.0,5.0,3.61)
rel_eb_vec=(0.02,0.01,0.005)
base_power=3.61

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def read_value(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def calculate_bd_rate(base_result_vec,test_result_vec):
    base_psnr=np.asarray([result["psnr"] for result in base_result_vec])
    test_psnr=np.asarray([result["psnr"] for result in test_result_vec])
    base_log_br=np.log(np.asarray([result["br"] for result in base_result_vec]))
    test_log_br=np.log(np.asarray([result["br"] for result in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    if not min_psnr<max_psnr:
        return math.nan,min_psnr,max_psnr
    base_polynomial=np.polyfit(base_psnr,base_log_br,2)
    test_polynomial=np.polyfit(test_psnr,test_log_br,2)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100,min_psnr,max_psnr

def benchmark(field,power,rel_eb,test_dir,prediction_structure_path,time_step=0):
    variable=f"{field}_t{time_step}_p{power:g}_{str(rel_eb).replace('.','_')}"
    os.symlink(os.path.join(data_dir,f"{field}.dat.{time_step}"),os.path.join(test_dir,f"{variable}.dat.{time_step}"))
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step",f"{{{time_step}}}","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-prediction-structure",prediction_structure_path,"-predictor","inverse_proportion","-actual-ref-num","10","-inverse-distance-power",str(power)]
    compression_output=run_command([compressor,*arguments])
    decompression_arguments=[]
    skip_next=False
    for argument in arguments:
        if skip_next:
            skip_next=False
            continue
        if argument=="-predictor":
            skip_next=True
            continue
        decompression_arguments.append(argument)
    decompression_output=run_command([decompressor,*decompression_arguments])
    result={
        "field":field,
        "time_step":time_step,
        "power":power,
        "rel_eb":rel_eb,
        "br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),
        "psnr":read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR"),
        "compression_time":read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),
        "decompression_time":read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),
    }
    decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(result["psnr"],decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{result['psnr']} != {decompressed_psnr}")
    return result

def main():
    result_vec=[]
    summary_vec=[]
    result_map={}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_ocean_idw_power_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        prediction_structure_path=os.path.join(test_dir,"prediction_structure.bin")
        for field in field_vec:
            result_map[field]={}
            for power in power_vec:
                power_result_vec=[]
                for rel_eb in rel_eb_vec:
                    result=benchmark(field,power,rel_eb,test_dir,prediction_structure_path)
                    result_vec.append(result)
                    power_result_vec.append(result)
                    print(f"{field} power={power:g},rel_eb={rel_eb}: br={result['br']:.6f},psnr={result['psnr']:.6f}",flush=True)
                result_map[field][power]=power_result_vec
            best_integer_power=None
            best_integer_bd_rate=math.inf
            for power in power_vec:
                bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map[field][base_power],result_map[field][power])
                summary_vec.append({"scope":field,"power":power,"bd_rate_vs_3_61":bd_rate,"min_psnr":min_psnr,"max_psnr":max_psnr,"integer_best":False})
                if power.is_integer() and bd_rate<best_integer_bd_rate:
                    best_integer_power=power
                    best_integer_bd_rate=bd_rate
                print(f"{field} BD-rate(power={power:g} vs 3.61)={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
            for summary in summary_vec:
                if summary["scope"]==field and summary["power"]==best_integer_power:
                    summary["integer_best"]=True
            print(f"{field} best integer power={best_integer_power:g},BD-rate={best_integer_bd_rate:.6f}%",flush=True)
    integer_power_vec=tuple(power for power in power_vec if power.is_integer())
    for power in integer_power_vec:
        value_vec=[summary["bd_rate_vs_3_61"] for summary in summary_vec if summary["scope"] in field_vec and summary["power"]==power]
        win_num=sum(summary["integer_best"] for summary in summary_vec if summary["scope"] in field_vec and summary["power"]==power)
        summary_vec.append({"scope":"all_fields_mean","power":power,"bd_rate_vs_3_61":statistics.fmean(value_vec),"min_psnr":math.nan,"max_psnr":math.nan,"integer_best":win_num})
        summary_vec.append({"scope":"all_fields_median","power":power,"bd_rate_vs_3_61":statistics.median(value_vec),"min_psnr":math.nan,"max_psnr":math.nan,"integer_best":win_num})
        print(f"all fields power={power:g}: mean BD-rate={statistics.fmean(value_vec):.6f}%,median={statistics.median(value_vec):.6f}%,best-field-num={win_num}",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    with open(summary_path,"w",newline="") as summary_file:
        writer=csv.DictWriter(summary_file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)
    print(f"逐点结果已写入{result_path}",flush=True)
    print(f"汇总结果已写入{summary_path}",flush=True)

if __name__=="__main__":
    main()
