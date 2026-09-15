import os,re,subprocess,tempfile

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
field="velocityZonal"

def read_value(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def main():
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_pointwise_oracle_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        os.symlink(os.path.join(data_dir,f"{field}.dat.0"),os.path.join(test_dir,f"{field}.dat.0"))
        arguments=["-data-dir",test_dir,"-variable",field,"-time-step","{0}","-M","REL","0.001","-reverse-mean-distance-only-order-experimental","-dag-cache",os.path.join(test_dir,"prediction_structure.bin"),"-predictor","inverse_proportion","-max-ref-num","10","-actual-ref-num","10","-inverse-distance-power","3.5","-d-only-cache-refill-threshold-experimental","8","-partition-num","4","-partition-adaptive-predictor-experimental","-partition-adaptive-sample-num-experimental","256","-partition-fitting-candidates-experimental","mixed","-pointwise-oracle-statistics-experimental"]
        result=subprocess.run([os.path.join(project_root,"build/s1_compress"),*arguments],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(result.stdout,flush=True)
        if result.returncode!=0:
            raise RuntimeError("逐点oracle实验失败")
        decompression_arguments=[argument for argument in arguments if argument!="-pointwise-oracle-statistics-experimental"]
        predictor_id=decompression_arguments.index("-predictor")
        del decompression_arguments[predictor_id:predictor_id+2]
        decompression_result=subprocess.run([os.path.join(project_root,"build/s1_decompress"),*decompression_arguments],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(decompression_result.stdout,flush=True)
        if decompression_result.returncode!=0:
            raise RuntimeError("逐点oracle实验的基线文件解压失败")
        baseline_br=read_value(r"^total cr= [^,]+, total br= ([^,]+),",result.stdout,"基线BR")
        baseline_psnr=read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",result.stdout,"基线PSNR")
        decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_result.stdout,"解压PSNR")
        if abs(baseline_psnr-decompressed_psnr)>1e-6:
            raise RuntimeError(f"压缩和解压PSNR不一致：{baseline_psnr} != {decompressed_psnr}")
        no_selector_br=read_value(r"^pointwise_oracle_no_selector_file_br=([^,]+),",result.stdout,"不含selector的oracle BR")
        with_selector_br=read_value(r"^pointwise_oracle_no_selector_file_br=[^,]+, with_selector_file_br=([^,]+),",result.stdout,"包含selector的oracle BR")
        print(f"基线BR={baseline_br:.6f}",flush=True)
        print(f"忽略selector的oracle BR={no_selector_br:.6f}，降低{(baseline_br-no_selector_br)/baseline_br*100:.3f}%",flush=True)
        print(f"单独压缩selector后的oracle BR={with_selector_br:.6f}，降低{(baseline_br-with_selector_br)/baseline_br*100:.3f}%",flush=True)

if __name__=="__main__":
    main()
