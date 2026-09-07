import csv,os,tempfile
from pathlib import Path
import t78_screen_tsunami_momentum_x_predictor as screen

screen.matrix_path=screen.project_root/"adjust_coord_matrices/Tsunami_momentum_x_spatial_prediction_ggd_ref4_candidate.txt"
result_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_matern32_screen.csv"
rel_eb_vec=(1e-3,1e-5)
config_vec=[]
for shape in (0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5):
    config_vec.append({"name":f"matern32_s{shape:g}_r10","shape":shape,"ref_num":10})
for ref_num in (4,6,8):
    config_vec.append({"name":f"matern32_s0.25_r{ref_num}","shape":0.25,"ref_num":ref_num})

def main():
    rows=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_tsunami_matern32_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(screen.data_dir/"coords.dat")
        (test_dir/"momentum_x.dat.0").symlink_to(screen.data_dir/"momentum_x.dat.0")
        prediction_structure_path=Path(temporary_dir)/"prediction_structure.bin"
        for config in config_vec:
            benchmark_config={"name":config["name"],"predictor":"rbf_fitting","power":3.6,"ref_num":config["ref_num"],"extra_arguments":["-global-order-before-round-split-experimental","-rbf-kernel","matern32","-rbf-shape",config["shape"]]}
            for rel_eb in rel_eb_vec:
                row={"kernel":"matern32","shape":config["shape"],**screen.benchmark(test_dir,prediction_structure_path,benchmark_config,rel_eb)}
                rows.append(row)
                print(f"{config['name']} REL={rel_eb}:BR={row['br']:.6f},PSNR={row['psnr']:.6f},MSE={row['prediction_mse']:.6g},compression={row['compression_time']:.6f} s,decompression={row['decompression_time']:.6f} s",flush=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for rel_eb in rel_eb_vec:
        current=sorted((row for row in rows if row["rel_eb"]==rel_eb),key=lambda row:row["br"])
        print(f"REL={rel_eb}最低BR候选：")
        for row in current[:6]:
            print(f"  {row['config']}:BR={row['br']:.6f},PSNR={row['psnr']:.6f},MSE={row['prediction_mse']:.6g}")
    print(f"结果={result_path}")

if __name__=="__main__":
    main()
