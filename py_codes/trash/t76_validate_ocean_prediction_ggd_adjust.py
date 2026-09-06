import sys,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.trash.t74_screen_ocean_spatial_adjust import benchmark,interpolate_log_br,write_csv

source_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean")
field_vec=("temperature","velocityZonal","zTop")
time_step_vec=(15,30)
config_vec=("original_coord","ref4_adjust","prediction_ggd")
rel_eb_vec=(1e-3,9e-4)
result_path=project_root/"py_codes/trash/results/ocean_spatial_adjust_prediction_ggd_heldout.csv"
summary_path=project_root/"py_codes/trash/results/ocean_spatial_adjust_prediction_ggd_heldout_summary.csv"

def main():
    result_vec=[]
    for field in field_vec:
        for time_step in time_step_vec:
            with tempfile.TemporaryDirectory(prefix=f"unsz_ggd_{field}_{time_step}_") as temporary_dir:
                test_dir=Path(temporary_dir)
                (test_dir/"coords.dat").symlink_to(source_dir/"coords.dat")
                (test_dir/f"{field}.dat.0").symlink_to(source_dir/f"{field}.dat.{time_step}")
                for config in config_vec:
                    prediction_structure_path=test_dir/f"prediction_{config}.bin"
                    for rel_eb in rel_eb_vec:
                        row=benchmark(field,config,test_dir,rel_eb,prediction_structure_path)
                        row["time_step"]=time_step
                        result_vec.append(row)
                        print(f"{field} t={time_step} {config} REL={rel_eb:g}:BR={row['br']:.9f},PSNR={row['psnr']:.6f}",flush=True)
    write_csv(result_path,result_vec)
    summary_vec=[]
    for field in field_vec:
        for time_step in time_step_vec:
            current_vec=[row for row in result_vec if row["field"]==field and row["time_step"]==time_step]
            original_vec=[row for row in current_vec if row["config"]=="original_coord"]
            target_psnr=next(row["psnr"] for row in original_vec if row["rel_eb"]==rel_eb_vec[0])
            original_br=interpolate_log_br(original_vec,target_psnr)
            ref4_br=interpolate_log_br([row for row in current_vec if row["config"]=="ref4_adjust"],target_psnr)
            prediction_ggd_br=interpolate_log_br([row for row in current_vec if row["config"]=="prediction_ggd"],target_psnr)
            summary_vec.append({"field":field,"time_step":time_step,"target_psnr":target_psnr,"original_coord_br":original_br,"ref4_adjust_br":ref4_br,"prediction_ggd_br":prediction_ggd_br,"prediction_ggd_vs_original_percent":(prediction_ggd_br/original_br-1)*100,"prediction_ggd_vs_ref4_percent":(prediction_ggd_br/ref4_br-1)*100})
    write_csv(summary_path,summary_vec)
    for row in summary_vec:
        print(f"{row['field']} t={row['time_step']}:码长loss相对原坐标={row['prediction_ggd_vs_original_percent']:.6f}%,相对二次方ref4={row['prediction_ggd_vs_ref4_percent']:.6f}%",flush=True)

if __name__=="__main__":
    main()
