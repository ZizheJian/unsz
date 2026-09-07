import csv,os,tempfile
from pathlib import Path
import numpy as np
import t78_screen_tsunami_momentum_x_predictor as screen

matrix_path=screen.project_root/"adjust_coord_matrices/Tsunami_momentum_x_spatial_prediction_ggd_ref4_candidate.txt"
result_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_prediction_matrix_rd.csv"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
config_vec=({"name":"idw_p3.6_r10_prediction_matrix","predictor":"inverse_proportion","power":3.6,"ref_num":10},{"name":"idw_p2_r4_prediction_matrix","predictor":"inverse_proportion","power":2,"ref_num":4},{"name":"hybrid_loo_r6_prediction_matrix","predictor":"hybrid_loo","power":3.6,"ref_num":6})

def curve(results):
    return {"br":np.asarray([float(result["br"]) for result in results]),"psnr":np.asarray([float(result["psnr"]) for result in results])}

def calculate_bd_rate(reference,test):
    lower=max(reference["psnr"].min(),test["psnr"].min())
    upper=min(reference["psnr"].max(),test["psnr"].max())
    reference_integral=np.polyint(np.polyfit(reference["psnr"],np.log(reference["br"]),3))
    test_integral=np.polyint(np.polyfit(test["psnr"],np.log(test["br"]),3))
    return (np.exp(((np.polyval(test_integral,upper)-np.polyval(test_integral,lower))-(np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)))/(upper-lower))-1)*100

def read_previous_results():
    previous_path=screen.project_root/"py_codes/trash/results/tsunami_momentum_x_predictor_rd.csv"
    with open(previous_path,newline="") as file:
        rows=list(csv.DictReader(file))
    return {name:curve([row for row in rows if row["config"]==name]) for name in ("idw_p2_r4","hybrid_loo_r6")}

def main():
    screen.matrix_path=matrix_path
    previous=read_previous_results()
    result_map={}
    all_results=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_tsunami_prediction_matrix_",dir=temporary_root) as temporary_dir:
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
                print(f"{config['name']} REL={rel_eb}:BR={result['br']:.6f},PSNR={result['psnr']:.6f}",flush=True)
            result_map[config["name"]]=results
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    for new_name,old_name in (("idw_p2_r4_prediction_matrix","idw_p2_r4"),("hybrid_loo_r6_prediction_matrix","hybrid_loo_r6")):
        print(f"{new_name}相对相同预测器的当前矩阵BD-rate={calculate_bd_rate(previous[old_name],curve(result_map[new_name])):.6f}%")
    print(f"结果={result_path}")

if __name__=="__main__":
    main()
