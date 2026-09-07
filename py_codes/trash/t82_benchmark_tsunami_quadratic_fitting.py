import csv,os,tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import t78_screen_tsunami_momentum_x_predictor as screen

matrix_path=screen.project_root/"adjust_coord_matrices/Tsunami_momentum_x_spatial_prediction_ggd_ref4_candidate.txt"
result_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_quadratic_fitting_rd.csv"
figure_path=screen.project_root/"figures/tsunami_momentum_x_quadratic_fitting_rd"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
config_vec=tuple({"name":f"quadratic_r{ref_num}","predictor":"quadratic_fitting","power":3.6,"ref_num":ref_num,"extra_arguments":["-global-order-before-round-split-experimental"]} for ref_num in (6,8,10))

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

def read_hybrid_curve():
    path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_combined_rd.csv"
    with open(path,newline="") as file:
        rows=[row for row in csv.DictReader(file) if row["config"]=="global_hybrid_loo_r6_prediction_matrix"]
    return curve(rows)

def main():
    screen.matrix_path=matrix_path
    existing=read_existing_curves()
    hybrid=read_hybrid_curve()
    result_map={}
    all_rows=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_tsunami_quadratic_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(screen.data_dir/"coords.dat")
        (test_dir/"momentum_x.dat.0").symlink_to(screen.data_dir/"momentum_x.dat.0")
        prediction_structure_path=Path(temporary_dir)/"prediction_structure.bin"
        for config in config_vec:
            rows=[]
            for rel_eb in rel_eb_vec:
                row=screen.benchmark(test_dir,prediction_structure_path,config,rel_eb)
                rows.append(row)
                all_rows.append(row)
                print(f"{config['name']} REL={rel_eb}:BR={row['br']:.6f},PSNR={row['psnr']:.6f},compression={row['compression_time']:.6f} s,decompression={row['decompression_time']:.6f} s",flush=True)
            result_map[config["name"]]=curve(rows)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    for config in config_vec:
        name=config["name"]
        rows=[row for row in all_rows if row["config"]==name]
        print(f"{name}:相对当前UNSZ={calculate_bd_rate(existing['UNSZ'],result_map[name]):.6f}%,相对UMC={calculate_bd_rate(existing['UMC'],result_map[name]):.6f}%,相对hybrid={calculate_bd_rate(hybrid,result_map[name]):.6f}%,平均压缩={np.mean([row['compression_time'] for row in rows]):.6f} s,平均解压={np.mean([row['decompression_time'] for row in rows]):.6f} s")
    figure,axis=plt.subplots(figsize=(7,5),layout="constrained")
    axis.plot(existing["UMC"]["br"],existing["UMC"]["psnr"],color="#D55E00",marker="o",linewidth=2,label="UMC")
    axis.plot(existing["UNSZ"]["br"],existing["UNSZ"]["psnr"],color="#0072B2",marker="s",linewidth=2,label="Current UNSZ")
    axis.plot(hybrid["br"],hybrid["psnr"],color="#009E73",marker="v",linewidth=2,label="Hybrid candidate")
    for config,color,marker in zip(config_vec,("#CC79A7","#56B4E9","#E69F00"),("^","P","D")):
        current=result_map[config["name"]]
        axis.plot(current["br"],current["psnr"],color=color,marker=marker,linewidth=2,label=config["name"])
    axis.set_xlim(left=0)
    axis.set_xlabel("Bit rate (bit/value)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Tsunami momentum_x quadratic fitting")
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
