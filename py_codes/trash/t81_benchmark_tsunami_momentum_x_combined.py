import csv,os,tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import t78_screen_tsunami_momentum_x_predictor as screen

matrix_path=screen.project_root/"adjust_coord_matrices/Tsunami_momentum_x_spatial_prediction_ggd_ref4_candidate.txt"
result_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_combined_rd.csv"
figure_path=screen.project_root/"figures/tsunami_momentum_x_optimized_rd"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
config_vec=({"name":"global_idw_p2_r4_prediction_matrix","predictor":"inverse_proportion","power":2,"ref_num":4,"extra_arguments":["-global-order-before-round-split-experimental"]},{"name":"global_hybrid_loo_r6_prediction_matrix","predictor":"hybrid_loo","power":3.6,"ref_num":6,"extra_arguments":["-global-order-before-round-split-experimental"]})

def curve(rows):
    return {"br":np.asarray([float(row["br"]) for row in rows]),"psnr":np.asarray([float(row["psnr"]) for row in rows])}

def calculate_bd_rate(reference,test):
    lower=max(reference["psnr"].min(),test["psnr"].min())
    upper=min(reference["psnr"].max(),test["psnr"].max())
    reference_integral=np.polyint(np.polyfit(reference["psnr"],np.log(reference["br"]),3))
    test_integral=np.polyint(np.polyfit(test["psnr"],np.log(test["br"]),3))
    return (np.exp(((np.polyval(test_integral,upper)-np.polyval(test_integral,lower))-(np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)))/(upper-lower))-1)*100

def read_curve(path,config):
    with open(path,newline="") as file:
        rows=[row for row in csv.DictReader(file) if row["config"]==config]
    return curve(rows)

def main():
    screen.matrix_path=matrix_path
    current_matrix_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_prediction_matrix_rd.csv"
    existing_csv=screen.data_dir/"momentum_x.csv"
    with open(existing_csv,newline="") as file:
        existing_rows=[row for row in csv.DictReader(file) if int(row["TIME_STEP_NUM"])==1 and float(row["REL_EB"]) in rel_eb_vec]
    existing_rows.sort(key=lambda row:float(row["REL_EB"]),reverse=True)
    unsz_curve={"br":np.asarray([float(row["UNSZ_BR"]) for row in existing_rows]),"psnr":np.asarray([float(row["UNSZ_PSNR"]) for row in existing_rows])}
    umc_curve={"br":np.asarray([float(row["UMC_BR"]) for row in existing_rows]),"psnr":np.asarray([float(row["UMC_PSNR"]) for row in existing_rows])}
    base_curve_map={"global_idw_p2_r4_prediction_matrix":read_curve(current_matrix_path,"idw_p2_r4_prediction_matrix"),"global_hybrid_loo_r6_prediction_matrix":read_curve(current_matrix_path,"hybrid_loo_r6_prediction_matrix")}
    result_map={}
    all_results=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_tsunami_combined_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(screen.data_dir/"coords.dat")
        (test_dir/"momentum_x.dat.0").symlink_to(screen.data_dir/"momentum_x.dat.0")
        prediction_structure_path=Path(temporary_dir)/"prediction_structure.bin"
        for config in config_vec:
            rows=[]
            for rel_eb in rel_eb_vec:
                result=screen.benchmark(test_dir,prediction_structure_path,config,rel_eb)
                rows.append(result)
                all_results.append(result)
                print(f"{config['name']} REL={rel_eb}:BR={result['br']:.6f},PSNR={result['psnr']:.6f}",flush=True)
            result_map[config["name"]]=curve(rows)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    for config in config_vec:
        name=config["name"]
        print(f"{name}:相对同预测器、不使用全局顺序BD-rate={calculate_bd_rate(base_curve_map[name],result_map[name]):.6f}%,相对当前UNSZ BD-rate={calculate_bd_rate(unsz_curve,result_map[name]):.6f}%,相对UMC BD-rate={calculate_bd_rate(umc_curve,result_map[name]):.6f}%")
    figure,axis=plt.subplots(figsize=(7,5),layout="constrained")
    axis.plot(umc_curve["br"],umc_curve["psnr"],color="#D55E00",marker="o",linewidth=2,label="UMC")
    axis.plot(unsz_curve["br"],unsz_curve["psnr"],color="#0072B2",marker="s",linewidth=2,label="Current UNSZ")
    axis.plot(result_map["global_idw_p2_r4_prediction_matrix"]["br"],result_map["global_idw_p2_r4_prediction_matrix"]["psnr"],color="#E69F00",marker="D",linewidth=2,label="Fast candidate")
    axis.plot(result_map["global_hybrid_loo_r6_prediction_matrix"]["br"],result_map["global_hybrid_loo_r6_prediction_matrix"]["psnr"],color="#009E73",marker="v",linewidth=2,label="Compression candidate")
    axis.set_xlim(left=0)
    axis.set_xlabel("Bit rate (bit/value)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Tsunami momentum_x optimized spatial prediction")
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
