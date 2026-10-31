import csv,math,subprocess
from pathlib import Path
import numpy as np

# 消融：不区分有效值和异常值，把-99999等异常值当作有效值，以单标签方式压缩
# 单标签时压缩器统计的PSNR会把异常值也算进去，因此写出解压结果，按标签配置的规则只在有效值上计算与论文相同的逐slice有效值PSNR
# 误差上限用ABS：REL乘以全部slice有效值的整体值域，使率失真曲线与完整UNSZ的PSNR范围重叠
project_root=Path(__file__).resolve().parents[2]
data_root=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data')
work_root=data_root/'ablation_work/exceptional_values_off'
output_dir=project_root/'results/ablation/exceptional_values_off/fields'
binary_dir=project_root/'install/bin'
fields=[('katrina_attr0','Katrina','attr0',162,lambda values:values>=-1000),('e3sm_h4_aoddust','E3SM-h4','AODDUST',720,lambda values:values>=-1000),('adcirc_haiyan_zeta','ADCIRC-Haiyan-1.5km','zeta',24,lambda values:values!=-99999)]
rel_ebs=(1e-1,1e-2,1e-3,1e-4,1e-5)


def valid_range(data_dir,field,slice_num,is_valid):
    minimum,maximum=math.inf,-math.inf
    for slice_id in range(slice_num):
        values=np.fromfile(data_dir/f'{field}.dat.{slice_id}',dtype=np.float32)
        valid=values[is_valid(values)]
        if valid.size:minimum,maximum=min(minimum,float(valid.min())),max(maximum,float(valid.max()))
    return maximum-minimum


def slice_valid_psnr(data_dir,stream_dir,field,slice_num,is_valid):
    # 与t1_start_c_code.py的psnr列相同：每个有效值的误差除以所在slice有效值的值域，再在全部有效值上平均；有效值为常量的slice不计误差
    error_sum,valid_num=0.0,0
    for slice_id in range(slice_num):
        original=np.fromfile(data_dir/f'{field}.dat.{slice_id}',dtype=np.float32).astype(np.float64)
        reconstructed=np.fromfile(stream_dir/f'{field}.dat.{slice_id}.unsz.bin',dtype=np.float32).astype(np.float64)
        valid=is_valid(original)
        valid_num+=int(valid.sum())
        if not valid.any():continue
        data_range=original[valid].max()-original[valid].min()
        if data_range>0:error_sum+=float((((reconstructed[valid]-original[valid])/data_range)**2).sum())
    return -10*math.log10(error_sum/valid_num) if error_sum>0 else math.inf


def main():
    output_dir.mkdir(parents=True,exist_ok=True)
    for task,dataset,field,slice_num,is_valid in fields:
        data_dir=data_root/dataset
        field_range=valid_range(data_dir,field,slice_num,is_valid)
        rows=[]
        for rel_eb in rel_ebs:
            stream_dir=work_root/task/f'rel_{rel_eb:g}'
            stream_dir.mkdir(parents=True,exist_ok=True)
            arguments=['-data-dir',str(data_dir),'-field',field,'-slice',f'[0,{slice_num-1}]','-coordinate-dim','3','-M','ABS',repr(rel_eb*field_range),'-output-dir',str(stream_dir)]
            for program in ('s1_compress','s1_decompress'):
                with (stream_dir/f'{program}.log').open('w')as log:
                    subprocess.run([str(binary_dir/program),*arguments],stdout=log,stderr=subprocess.STDOUT,check=True)
            compressed_bytes=sum((stream_dir/f'{field}.dat.{slice_id}.unsz').stat().st_size for slice_id in range(slice_num))
            point_num=(data_dir/f'{field}.dat.0').stat().st_size//4
            br=8*compressed_bytes/(point_num*slice_num)
            psnr=slice_valid_psnr(data_dir,stream_dir,field,slice_num,is_valid)
            rows.append({'task':task,'field':field,'compressor':'unsz','rel_eb':rel_eb,'abs_eb':rel_eb*field_range,'br':br,'psnr':psnr,'compressed_bytes':compressed_bytes})
            print(f'{task} REL={rel_eb:g} br={br:.6f} psnr={psnr:.4f}',flush=True)
        with (output_dir/f'{task}.csv').open('w',newline='')as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
