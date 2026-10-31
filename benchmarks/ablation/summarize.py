import csv,math,sys
from pathlib import Path

# 汇总消融结果：每个field在两条率失真曲线共同覆盖的PSNR范围内求码率比（消融/完整UNSZ）的几何平均，与论文中比较UNSZ和UMC的方法相同
project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.summarize_current_results import geometric_metric_ratio_at_same_psnr
csv.field_size_limit(1<<30)
excluded_fields={'RMS_T'}
variants=('rescaling_off','tensor_prediction_off','exceptional_values_off','spatial_residual_off','eb_schedule_off','local_ordering','idw_weights','forward_dag')


def read_unsz_rows(paths):
    groups={}
    for path in paths:
        for row in csv.DictReader(path.open()):
            if row['compressor']=='unsz' and row['field'] not in excluded_fields:
                groups.setdefault((row['task'],row['field']),[]).append(row)
    return {key:sorted(rows,key=lambda row:float(row['psnr'])) for key,rows in groups.items()}


def main():
    baseline=read_unsz_rows(sorted((project_root/'results/current/fields').glob('*.csv')))
    summary=[]
    for variant in variants:
        directory=project_root/'results/ablation'/variant/'fields'
        if not directory.is_dir():continue
        ablated=read_unsz_rows(sorted(directory.glob('*.csv')))
        ratios={}
        for key,rows in sorted(ablated.items()):
            if len(rows)<2:
                print(f'{variant} {key}: 只有{len(rows)}个点，跳过');continue
            ratios[key]=geometric_metric_ratio_at_same_psnr(baseline[key],rows,'br')
            summary.append({'variant':variant,'task':key[0],'field':key[1],'bit_rate_ratio':ratios[key]})
        if not ratios:continue
        affected_mean=math.exp(sum(math.log(value)for value in ratios.values())/len(ratios))
        all_mean=math.exp(sum(math.log(value)for value in ratios.values())/len(baseline))
        print(f'{variant}: {len(ratios)}个field，受影响field上码率增加到{affected_mean:.3f}倍，全部{len(baseline)}个field上{all_mean:.3f}倍；范围{min(ratios.values()):.3f}～{max(ratios.values()):.3f}')
        summary.append({'variant':variant,'task':'mean_over_affected_fields','field':len(ratios),'bit_rate_ratio':affected_mean})
        summary.append({'variant':variant,'task':'mean_over_all_fields','field':len(baseline),'bit_rate_ratio':all_mean})
    with (project_root/'results/ablation/summary.csv').open('w',newline='')as file:
        writer=csv.DictWriter(file,fieldnames=['variant','task','field','bit_rate_ratio'],lineterminator='\n')
        writer.writeheader();writer.writerows(summary)


if __name__=='__main__':
    main()
