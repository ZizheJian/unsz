import json,re,statistics,subprocess
from pathlib import Path

# 比较三种预测顺序的预处理耗时：正式的反向MSD消去(global)、插值式分轮排序(local)和正向构造(forward)
# 每个数据集取一个代表field，REL 1e-3，三种顺序交替串行运行，只运行压缩端，耗时取程序内部计时
project_root=Path(__file__).resolve().parents[2]
data_root=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data')
install_dir=project_root/'install_ablation'
work_root=data_root/'ablation_work/ordering_timing'
output_path=project_root/'results/ablation/ordering_timing/times.json'
label_dir=project_root/'label_configs'
cases={
    'tpost20_scalar':['-data-dir',str(data_root/'Tpost20'),'-field','scalar','-slice','[0,19]','-M','REL','1e-3'],
    'katrina_attr0':['-data-dir',str(data_root/'Katrina'),'-field','attr0','-slice','[0,161]','-label-config',str(label_dir/'Katrina_attr0_rbf_rel_1e-3.json')],
    'e3sm_h4_mass_dst_srf':['-data-dir',str(data_root/'E3SM-h4'),'-field','Mass_dst_srf','-slice','[0,719]','-M','REL','1e-3'],
    'ocean_salinity':['-data-dir',str(data_root/'Ocean'),'-field','salinity','-slice','[0,30]','-M','REL','1e-3'],
    'adcirc_haiyan_depth':['-data-dir',str(data_root/'ADCIRC-Haiyan-1.5km'),'-field','depth','-slice','[0,0]','-M','REL','1e-3'],
    'les_velocity_x':['-data-dir',str(data_root/'LES'),'-field','velocity_x','-slice','[0,0]','-M','REL','1e-3'],
    'taps_temperature':['-data-dir',str(data_root/'TAPS'),'-field','TEMPERATURE','-slice','[0,0]','-M','REL','1e-3'],
}
orderings=('global','local','forward')
repeat_num=3
patterns={'dag_time':r'^prediction DAG and weights preprocessing time= ([0-9.eE+-]+) s$','preprocessing_time':r'^total compression preprocessing time excluding coordinate and field-data input= ([0-9.eE+-]+) s$','compression_time':r'^total compression time= ([0-9.eE+-]+) s$','br':r'^total cr= [^,]+, total br= ([0-9.eE+-]+),'}


def main():
    output_path.parent.mkdir(parents=True,exist_ok=True)
    times={case:{ordering:{key:[] for key in patterns} for ordering in orderings} for case in cases}
    for repeat in range(repeat_num):
        for case,arguments in cases.items():
            for ordering in orderings if repeat%2==0 else orderings[::-1]:
                out=work_root/case/ordering
                out.mkdir(parents=True,exist_ok=True)
                text=subprocess.run([str(install_dir/'bin/s1_compress'),*arguments,'-coordinate-dim','3','-output-dir',str(out),'-ordering',ordering],capture_output=True,text=True,check=True).stdout
                for key,pattern in patterns.items():
                    times[case][ordering][key].append(float(re.search(pattern,text,re.MULTILINE).group(1)))
            print(f'repeat {repeat} {case} done',flush=True)
            output_path.write_text(json.dumps(times,indent=1)+'\n')
    print('case,ordering,dag_time_median_s,preprocessing_time_median_s,br')
    for case in cases:
        for ordering in orderings:
            item=times[case][ordering]
            print(f'{case},{ordering},{statistics.median(item["dag_time"]):.3f},{statistics.median(item["preprocessing_time"]):.3f},{item["br"][0]:.6f}')


if __name__=='__main__':
    main()
