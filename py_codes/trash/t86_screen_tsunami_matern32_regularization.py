import csv,os,tempfile
from pathlib import Path
import t78_screen_tsunami_momentum_x_predictor as screen

screen.matrix_path=screen.project_root/"adjust_coord_matrices/Tsunami_momentum_x_spatial_prediction_ggd_ref4_candidate.txt"
result_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_matern32_regularization_screen.csv"
rel_eb_vec=(1e-3,1e-5)
shape_vec=(0.01,0.02,0.03,0.05,0.075,0.1)
nugget_vec=(1e-10,1e-8,1e-6,1e-4)

def main():
    rows=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_tsunami_matern32_regularization_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(screen.data_dir/"coords.dat")
        (test_dir/"momentum_x.dat.0").symlink_to(screen.data_dir/"momentum_x.dat.0")
        prediction_structure_path=Path(temporary_dir)/"prediction_structure.bin"
        for shape in shape_vec:
            for nugget in nugget_vec:
                name=f"matern32_s{shape:g}_n{nugget:g}_r10"
                config={"name":name,"predictor":"rbf_fitting","power":3.6,"ref_num":10,"extra_arguments":["-global-order-before-round-split-experimental","-rbf-kernel","matern32","-rbf-shape",shape,"-rbf-nugget",nugget]}
                for rel_eb in rel_eb_vec:
                    row={"kernel":"matern32","shape":shape,"nugget":nugget,**screen.benchmark(test_dir,prediction_structure_path,config,rel_eb)}
                    rows.append(row)
                    print(f"{name} REL={rel_eb}:BR={row['br']:.6f},PSNR={row['psnr']:.6f},MSE={row['prediction_mse']:.6g}",flush=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for rel_eb in rel_eb_vec:
        current=sorted((row for row in rows if row["rel_eb"]==rel_eb),key=lambda row:row["br"])
        print(f"REL={rel_eb}最低BR候选：")
        for row in current[:8]:
            print(f"  {row['config']}:BR={row['br']:.6f},PSNR={row['psnr']:.6f},MSE={row['prediction_mse']:.6g}")
    print(f"结果={result_path}")

if __name__=="__main__":
    main()
