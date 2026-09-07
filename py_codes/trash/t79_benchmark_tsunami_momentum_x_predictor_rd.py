import csv,os,tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import t78_screen_tsunami_momentum_x_predictor as screen

result_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_predictor_rd.csv"
figure_path=screen.project_root/"figures/tsunami_momentum_x_predictor_rd"
source_csv_path=screen.data_dir/"momentum_x.csv"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
config_vec=({"name":"idw_p2_r4","predictor":"inverse_proportion","power":2,"ref_num":4},{"name":"fitting_r6","predictor":"fitting","power":3.6,"ref_num":6},{"name":"hybrid_loo_r6","predictor":"hybrid_loo","power":3.6,"ref_num":6})

def read_existing_curves():
    with open(source_csv_path,newline="") as file:
        rows=[row for row in csv.DictReader(file) if int(row["TIME_STEP_NUM"])==1 and float(row["REL_EB"]) in rel_eb_vec]
    rows.sort(key=lambda row:float(row["REL_EB"]),reverse=True)
    if len(rows)!=len(rel_eb_vec):
        raise RuntimeError(f"现有Tsunami CSV中应有{len(rel_eb_vec)}个点，实际为{len(rows)}")
    return {name:{"br":np.asarray([float(row[f"{name}_BR"]) for row in rows]),"psnr":np.asarray([float(row[f"{name}_PSNR"]) for row in rows])} for name in ("UNSZ","UMC")}

def curve(results):
    return {"br":np.asarray([result["br"] for result in results]),"psnr":np.asarray([result["psnr"] for result in results])}

def calculate_bd_rate(reference,test):
    lower=max(reference["psnr"].min(),test["psnr"].min())
    upper=min(reference["psnr"].max(),test["psnr"].max())
    reference_integral=np.polyint(np.polyfit(reference["psnr"],np.log(reference["br"]),3))
    test_integral=np.polyint(np.polyfit(test["psnr"],np.log(test["br"]),3))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def main():
    existing=read_existing_curves()
    result_map={}
    all_results=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_tsunami_predictor_rd_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(screen.data_dir/"coords.dat")
        (test_dir/"momentum_x.dat.0").symlink_to(screen.data_dir/"momentum_x.dat.0")
        prediction_structure_path=Path(temporary_dir)/"prediction_structure.bin"
        for config in config_vec:
            results=[]
            for rel_eb in rel_eb_vec:
                result=screen.benchmark(test_dir,prediction_structure_path,config,rel_eb)
                results.append(result)
                all_results.append(result)
                print(f"{config['name']} REL={rel_eb}:BR={result['br']:.6f},PSNR={result['psnr']:.6f},compression={result['compression_time']:.6f} s,decompression={result['decompression_time']:.6f} s",flush=True)
            result_map[config["name"]]=results
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    figure,axis=plt.subplots(figsize=(7,5),layout="constrained")
    axis.plot(existing["UMC"]["br"],existing["UMC"]["psnr"],color="#D55E00",marker="o",linewidth=2,label="UMC")
    axis.plot(existing["UNSZ"]["br"],existing["UNSZ"]["psnr"],color="#0072B2",marker="s",linewidth=2,label="UNSZ IDW d^-3.6,10 refs")
    style_vec=(("#CC79A7","^"),("#E69F00","D"),("#009E73","v"))
    for config,style in zip(config_vec,style_vec):
        current_curve=curve(result_map[config["name"]])
        axis.plot(current_curve["br"],current_curve["psnr"],color=style[0],marker=style[1],linewidth=2,label=config["name"])
        print(f"{config['name']}:相对当前UNSZ BD-rate={calculate_bd_rate(existing['UNSZ'],current_curve):.6f}%,相对UMC BD-rate={calculate_bd_rate(existing['UMC'],current_curve):.6f}%,平均压缩时间={np.mean([result['compression_time'] for result in result_map[config['name']]]):.6f} s,平均解压时间={np.mean([result['decompression_time'] for result in result_map[config['name']]]):.6f} s")
    axis.set_xlim(left=0)
    axis.set_xlabel("Bit rate (bit/value)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Tsunami momentum_x predictor comparison")
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
