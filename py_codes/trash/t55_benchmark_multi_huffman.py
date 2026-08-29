import argparse,csv,math,os,re,subprocess,tempfile
import t44_benchmark_ocean_idw_power as common

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
data_dir=os.path.join(data_root,"Ocean")
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"ocean_multi_huffman_benchmark.csv")
summary_path=os.path.join(project_root,"ocean_multi_huffman_summary.csv")
field_vec=common.field_vec
block_num_vec=(2,4)
mode_vec=("none","round","block","block_round")
rel_eb=0.001

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

def benchmark(field,block_num,mode,test_dir,prediction_structure_path,adjust_coord=False):
    variable=field
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step","{0}","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-prediction-structure",prediction_structure_path,"-predictor","inverse_proportion","-max-ref-num","10","-actual-ref-num","10","-inverse-distance-power","3.5","-d-only-cache-refill-threshold-experimental","8","-independent-block-num-experimental",str(block_num),"-multi-huffman-experimental",mode]
    if adjust_coord:
        arguments.append("-adjust-coord")
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
    psnr=read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    return {"field":field,"block_num":block_num,"mode":mode,"rel_eb":rel_eb,"br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"value_huffman_bytes":int(read_value(r"^value_huffman_bytes=(\d+),",compression_output,"Huffman字节数")),"value_zstd_bytes":int(read_value(r"^value_huffman_bytes=\d+, value_zstd_bytes=(\d+),",compression_output,"Zstd字节数")),"file_bytes":int(read_value(r"^value_huffman_bytes=\d+, value_zstd_bytes=\d+, file_bytes=(\d+)$",compression_output,"文件字节数"))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--katrina-smoke",action="store_true")
    args=parser.parse_args()
    source_data_dir=os.path.join(data_root,"Katrina") if args.katrina_smoke else data_dir
    test_field_vec=("attr0",) if args.katrina_smoke else (field_vec[:1] if args.smoke else field_vec)
    test_block_num_vec=block_num_vec[:1] if args.smoke or args.katrina_smoke else block_num_vec
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_multi_huffman_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(source_data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for field in test_field_vec:
            os.symlink(os.path.join(source_data_dir,f"{field}.dat.0"),os.path.join(test_dir,f"{field}.dat.0"))
        for block_num in test_block_num_vec:
            prediction_structure_path=os.path.join(test_dir,f"prediction_structure_b{block_num}.bin")
            for field in test_field_vec:
                for mode in mode_vec:
                    result=benchmark(field,block_num,mode,test_dir,prediction_structure_path,args.katrina_smoke)
                    result_vec.append(result)
                    print(f"{field} block_num={block_num},mode={mode}: br={result['br']:.9f},psnr={result['psnr']:.6f},huffman={result['value_huffman_bytes']},zstd={result['value_zstd_bytes']},file={result['file_bytes']}",flush=True)
    summary_vec=[]
    for block_num in test_block_num_vec:
        for field in test_field_vec:
            field_result_vec=[result for result in result_vec if result["block_num"]==block_num and result["field"]==field]
            baseline=next(result for result in field_result_vec if result["mode"]=="none")
            for result in field_result_vec:
                summary_vec.append({"field":field,"block_num":block_num,"mode":result["mode"],"br_reduction_vs_single_huffman":(baseline["br"]-result["br"])/baseline["br"]*100,"file_byte_reduction_vs_single_huffman":(baseline["file_bytes"]-result["file_bytes"])/baseline["file_bytes"]*100})
    if args.smoke or args.katrina_smoke:
        print("multi-Huffman smoke test passed",flush=True)
        return
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    with open(summary_path,"w",newline="") as summary_file:
        writer=csv.DictWriter(summary_file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)
    print(f"逐项结果已写入{result_path}",flush=True)
    print(f"汇总结果已写入{summary_path}",flush=True)

if __name__=="__main__":
    main()
