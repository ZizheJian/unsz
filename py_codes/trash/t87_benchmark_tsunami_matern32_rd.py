import csv,os,tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import t78_screen_tsunami_momentum_x_predictor as screen
import t82_benchmark_tsunami_quadratic_fitting as quadratic

screen.matrix_path=screen.project_root/"adjust_coord_matrices/Tsunami_momentum_x_spatial_prediction_ggd_ref4_candidate.txt"
result_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_matern32_rd.csv"
figure_path=screen.project_root/"figures/tsunami_momentum_x_matern32_rd"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
shape_vec=(0.03,0.05,0.075)

def curve(rows):
    return {"br":np.asarray([float(row["br"]) for row in rows]),"psnr":np.asarray([float(row["psnr"]) for row in rows])}

def read_quadratic_curve():
    with open(screen.project_root/"py_codes/trash/results/tsunami_momentum_x_quadratic_fitting_rd.csv",newline="") as file:
        rows=[row for row in csv.DictReader(file) if row["config"]=="quadratic_r10"]
    return curve(rows)

def main():
    rows=[]
    result_map={}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_tsunami_matern32_rd_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(screen.data_dir/"coords.dat")
        (test_dir/"momentum_x.dat.0").symlink_to(screen.data_dir/"momentum_x.dat.0")
        prediction_structure_path=Path(temporary_dir)/"prediction_structure.bin"
        for shape in shape_vec:
            name=f"matern32_s{shape:g}_n1e-6_r10"
            config={"name":name,"predictor":"rbf_fitting","power":3.6,"ref_num":10,"extra_arguments":["-global-order-before-round-split-experimental","-rbf-kernel","matern32","-rbf-shape",shape,"-rbf-nugget",1e-6]}
            current=[]
            for rel_eb in rel_eb_vec:
                row={"kernel":"matern32","shape":shape,"nugget":1e-6,**screen.benchmark(test_dir,prediction_structure_path,config,rel_eb)}
                rows.append(row)
                current.append(row)
                print(f"{name} REL={rel_eb}:BR={row['br']:.6f},PSNR={row['psnr']:.6f},compression={row['compression_time']:.6f} s,decompression={row['decompression_time']:.6f} s",flush=True)
            result_map[name]=curve(current)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    existing=quadratic.read_existing_curves()
    quadratic_curve=read_quadratic_curve()
    for shape in shape_vec:
        name=f"matern32_s{shape:g}_n1e-6_r10"
        current_rows=[row for row in rows if row["config"]==name]
        print(f"{name}:相对当前UNSZ={quadratic.calculate_bd_rate(existing['UNSZ'],result_map[name]):.6f}%,相对UMC={quadratic.calculate_bd_rate(existing['UMC'],result_map[name]):.6f}%,相对二次拟合={quadratic.calculate_bd_rate(quadratic_curve,result_map[name]):.6f}%,平均压缩={np.mean([row['compression_time'] for row in current_rows]):.6f} s,平均解压={np.mean([row['decompression_time'] for row in current_rows]):.6f} s")
    figure,axis=plt.subplots(figsize=(7,5),layout="constrained")
    axis.plot(existing["UMC"]["br"],existing["UMC"]["psnr"],color="#999999",marker="^",linewidth=2,label="UMC")
    axis.plot(quadratic_curve["br"],quadratic_curve["psnr"],color="#0072B2",marker="s",linewidth=2,label="Quadratic fitting")
    for shape,color,marker in zip(shape_vec,("#D55E00","#009E73","#CC79A7"),("o","D","v")):
        name=f"matern32_s{shape:g}_n1e-6_r10"
        current=result_map[name]
        axis.plot(current["br"],current["psnr"],color=color,marker=marker,linewidth=2,label=f"Matérn-3/2 shape={shape:g}")
    axis.set_xlim(left=0)
    axis.set_xlabel("Bit rate (bit/value)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Tsunami momentum_x Matérn-3/2 RBF")
    axis.grid(True,linestyle=":",alpha=0.55)
    axis.legend(frameon=False,loc="lower right")
    figure_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(figure_path.with_suffix(".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(figure_path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)
    print(f"结果={result_path}")
    print(f"图像={figure_path.with_suffix('.png')}")

if __name__=="__main__":
    main()
