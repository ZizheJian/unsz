import csv,os,tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import t78_screen_tsunami_momentum_x_predictor as screen

result_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_quadratic_matrix_ablation.csv"
figure_path=screen.project_root/"figures/tsunami_momentum_x_quadratic_matrix_ablation"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
matrix_config_vec=(
    {"name":"second_power_ref4","path":screen.project_root/"adjust_coord_matrices/Tsunami_momentum_x_spatial.txt"},
    {"name":"prediction_ggd_ref4","path":screen.project_root/"adjust_coord_matrices/Tsunami_momentum_x_spatial_prediction_ggd_ref4_candidate.txt"},
)
predictor_config={"name":"quadratic_r10","predictor":"quadratic_fitting","power":3.6,"ref_num":10,"extra_arguments":["-global-order-before-round-split-experimental"]}

def curve(rows):
    return {"br":np.asarray([float(row["br"]) for row in rows]),"psnr":np.asarray([float(row["psnr"]) for row in rows])}

def calculate_bd_rate(reference,test):
    lower=max(reference["psnr"].min(),test["psnr"].min())
    upper=min(reference["psnr"].max(),test["psnr"].max())
    reference_integral=np.polyint(np.polyfit(reference["psnr"],np.log(reference["br"]),3))
    test_integral=np.polyint(np.polyfit(test["psnr"],np.log(test["br"]),3))
    return (np.exp(((np.polyval(test_integral,upper)-np.polyval(test_integral,lower))-(np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)))/(upper-lower))-1)*100

def read_existing_curves():
    with open(screen.data_dir/"momentum_x.csv",newline="") as file:
        rows=[row for row in csv.DictReader(file) if int(row["TIME_STEP_NUM"])==1 and float(row["REL_EB"]) in rel_eb_vec]
    rows.sort(key=lambda row:float(row["REL_EB"]),reverse=True)
    return {name:{"br":np.asarray([float(row[f"{name}_BR"]) for row in rows]),"psnr":np.asarray([float(row[f"{name}_PSNR"]) for row in rows])} for name in ("UNSZ","UMC")}

def main():
    result_map={}
    all_rows=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_tsunami_quadratic_matrix_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(screen.data_dir/"coords.dat")
        (test_dir/"momentum_x.dat.0").symlink_to(screen.data_dir/"momentum_x.dat.0")
        for matrix_config in matrix_config_vec:
            screen.matrix_path=matrix_config["path"]
            prediction_structure_path=Path(temporary_dir)/f"{matrix_config['name']}_prediction_structure.bin"
            rows=[]
            for rel_eb in rel_eb_vec:
                row=screen.benchmark(test_dir,prediction_structure_path,predictor_config,rel_eb)
                row={"matrix":matrix_config["name"],**row}
                rows.append(row)
                all_rows.append(row)
                print(f"{matrix_config['name']} REL={rel_eb}:BR={row['br']:.6f},PSNR={row['psnr']:.6f},compression={row['compression_time']:.6f} s,decompression={row['decompression_time']:.6f} s",flush=True)
            result_map[matrix_config["name"]]=curve(rows)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    bd_rate=calculate_bd_rate(result_map["second_power_ref4"],result_map["prediction_ggd_ref4"])
    print(f"预测码长loss矩阵相对二次方loss矩阵BD-rate={bd_rate:.6f}%")
    existing=read_existing_curves()
    for matrix_config in matrix_config_vec:
        name=matrix_config["name"]
        print(f"{name}相对当前UNSZ={calculate_bd_rate(existing['UNSZ'],result_map[name]):.6f}%,相对UMC={calculate_bd_rate(existing['UMC'],result_map[name]):.6f}%")
    figure,axis=plt.subplots(figsize=(7,5),layout="constrained")
    axis.plot(existing["UMC"]["br"],existing["UMC"]["psnr"],color="#999999",marker="^",linewidth=2,label="UMC")
    for matrix_config,color,marker in zip(matrix_config_vec,("#0072B2","#D55E00"),("s","o")):
        current=result_map[matrix_config["name"]]
        axis.plot(current["br"],current["psnr"],color=color,marker=marker,linewidth=2,label=matrix_config["name"])
    axis.set_xlim(left=0)
    axis.set_xlabel("Bit rate (bit/value)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Tsunami momentum_x quadratic fitting matrix ablation")
    axis.grid(True,linestyle=":",alpha=0.55)
    axis.legend(frameon=False)
    figure_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(figure_path.with_suffix(".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(figure_path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)
    print(f"结果={result_path}")
    print(f"图像={figure_path.with_suffix('.png')}")

if __name__=="__main__":
    main()
