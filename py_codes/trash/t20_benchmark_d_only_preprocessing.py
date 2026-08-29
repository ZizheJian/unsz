import hashlib,os,re,subprocess,tempfile

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"install/bin/s1_compress")
decompressor=os.path.join(project_root,"install/bin/s1_decompress")
time_step_num=5
dataset_vec=(
    {"name":"Katrina","variable":"attr0","adjust_coord":True},
    {"name":"Ocean","variable":"temperature","adjust_coord":False},
)

def read_value(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return match.group(1).strip()

def run_dataset(dataset):
    source_dir=os.path.join(data_root,dataset["name"])
    with tempfile.TemporaryDirectory(prefix=f"unsz_d_only_{dataset['name']}_") as test_dir:
        os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for time_step in range(time_step_num):
            os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}"),os.path.join(test_dir,f"{dataset['variable']}.dat.{time_step}"))
        arguments=["-data-dir",test_dir,"-variable",dataset["variable"],"-time-step",f"[0,{time_step_num-1}]","-M","REL","0.01","-reverse-mean-distance-only-order-experimental"]
        if dataset["adjust_coord"]:
            arguments.append("-adjust-coord")
        compression_result=subprocess.run([compressor,*arguments],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if compression_result.returncode!=0:
            print(compression_result.stdout,flush=True)
            raise RuntimeError(f"压缩命令失败：{compression_result.returncode}")
        compression=compression_result.stdout
        compressed_hash=hashlib.sha256()
        for time_step in range(time_step_num):
            with open(os.path.join(test_dir,f"{dataset['variable']}.dat.{time_step}.unsz"),"rb") as compressed_file:
                compressed_hash.update(compressed_file.read())
        decompression_result=subprocess.run([decompressor,*arguments],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if decompression_result.returncode!=0:
            print(decompression_result.stdout,flush=True)
            raise RuntimeError(f"解压命令失败：{decompression_result.returncode}")
        decompression=decompression_result.stdout
        preprocessing=read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression,"预处理时间")
        sorting=read_value(r"^Target sorting time: ([^ ]+) s$",compression,"目标排序时间")
        reference_generation=read_value(r"^Prediction reference generation time: ([^ ]+) s$",compression,"参考结构生成时间")
        cache_hit_num=read_value(r"^D-only score cache: ([0-9]+) hits, [0-9]+ misses$",compression,"D分数缓存命中数")
        cache_miss_num=read_value(r"^D-only score cache: [0-9]+ hits, ([0-9]+) misses$",compression,"D分数缓存未命中数")
        br=read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression,"总BR")
        psnr=read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression,"总PSNR")
        decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression,"解压总PSNR")
        cache_hit_ratio=int(cache_hit_num)/(int(cache_hit_num)+int(cache_miss_num))
        print(f"{dataset['name']}: preprocessing={preprocessing} s, sorting={sorting} s, reference_generation={reference_generation} s, cache_hit_ratio={cache_hit_ratio:.6f}, br={br}, psnr={psnr}, decompressed_psnr={decompressed_psnr}, sha256={compressed_hash.hexdigest()}",flush=True)

def main():
    for dataset in dataset_vec:
        run_dataset(dataset)

if __name__=="__main__":
    main()
