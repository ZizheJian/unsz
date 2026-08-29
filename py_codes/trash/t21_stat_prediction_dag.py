import os,re,subprocess,tempfile

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"install/bin/s1_compress")
dataset_vec=(
    {"name":"Katrina","variable":"attr0","adjust_coord":True},
    {"name":"Ocean","variable":"temperature","adjust_coord":False},
)

def main():
    for dataset in dataset_vec:
        source_dir=os.path.join(data_root,dataset["name"])
        with tempfile.TemporaryDirectory(prefix=f"unsz_dag_{dataset['name']}_") as test_dir:
            os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
            os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.0"),os.path.join(test_dir,f"{dataset['variable']}.dat.0"))
            arguments=[compressor,"-data-dir",test_dir,"-variable",dataset["variable"],"-time-step","[0,0]","-M","REL","0.01","-reverse-mean-distance-only-order-experimental","-parallelism-stats"]
            if dataset["adjust_coord"]:
                arguments.append("-adjust-coord")
            result=subprocess.run(arguments,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if result.returncode!=0:
                print(result.stdout,flush=True)
                raise RuntimeError(f"{dataset['name']}压缩失败：{result.returncode}")
            match=re.search(r"Prediction DAG statistics:\n(.*?)preprocessing time excluding adjust coord=",result.stdout,re.DOTALL)
            if not match:
                raise RuntimeError(f"{dataset['name']}输出中缺少预测DAG统计")
            print(dataset["name"],flush=True)
            print(match.group(1).strip(),flush=True)

if __name__=="__main__":
    main()
