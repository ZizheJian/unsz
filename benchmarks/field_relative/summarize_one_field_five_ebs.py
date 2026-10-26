import argparse,csv,json,math
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
expected_rel_ebs=(1e-1,1e-2,1e-3,1e-4,1e-5)

def load_curve(root,codec):
    points=[]
    for rel_eb in expected_rel_ebs:
        path=root/codec/f'rel_{rel_eb:g}_t1'/'summary.json'
        if not path.is_file():raise FileNotFoundError(path)
        result=json.loads(path.read_text())
        if not result.get('valid')or result.get('field_num')!=1:raise RuntimeError(f'无效结果：{path}')
        points.append({'codec':codec,'rel_eb':rel_eb,'br':float(result['br']),'psnr':float(result['normalized_psnr']),'compression_time':float(result['compression_time']),'decompression_time':float(result['decompression_time']),'max_error_over_eb':float(result['max_error_over_eb'])})
    points.sort(key=lambda point:point['psnr'])
    if any(points[index]['psnr']>=points[index+1]['psnr']or points[index]['br']>=points[index+1]['br']for index in range(len(points)-1)):raise RuntimeError(f'{codec}的RD点不是严格单调的')
    return points

def interpolate_log_rate(points,psnr):
    for left,right in zip(points,points[1:]):
        if left['psnr']<=psnr<=right['psnr']:
            weight=(psnr-left['psnr'])/(right['psnr']-left['psnr'])
            return math.log(left['br'])+weight*(math.log(right['br'])-math.log(left['br']))
    raise RuntimeError(f'PSNR={psnr}不在曲线范围内')

def matched_quality_difference(reference,candidate):
    lower=max(reference[0]['psnr'],candidate[0]['psnr'])
    upper=min(reference[-1]['psnr'],candidate[-1]['psnr'])
    if not lower<upper:raise RuntimeError('两条RD曲线没有重叠的PSNR区间')
    knots=sorted({lower,upper,*[point['psnr']for point in reference if lower<point['psnr']<upper],*[point['psnr']for point in candidate if lower<point['psnr']<upper]})
    differences=[interpolate_log_rate(candidate,psnr)-interpolate_log_rate(reference,psnr)for psnr in knots]
    integral=sum((differences[index]+differences[index+1])*(knots[index+1]-knots[index])/2 for index in range(len(knots)-1))
    average=integral/(upper-lower)
    return lower,upper,100*(math.exp(average)-1),knots,differences

def load_slice_relative_field_curve(codec,field):
    points=[]
    unsz_directories={1e-1:'rel_1e-1',1e-2:'rel_1e-2',1e-3:'sparse_window',1e-4:'rel_1e-4',1e-5:'rel_1e-5'}
    for rel_eb in expected_rel_ebs:
        if codec=='unsz':
            path=project_root/'results/ocean_large_tensor_full'/unsz_directories[rel_eb]/f'{field}.csv'
            result=next(csv.DictReader(path.open()))
            point={'codec':codec,'rel_eb':rel_eb,'br':float(result['br']),'psnr':float(result['psnr'])}
        else:
            path=project_root/'results/ocean_large_tensor_1d'/codec/f'rel_{rel_eb:g}_t1'/f'{field}.json'
            result=json.loads(path.read_text())
            point={'codec':codec,'rel_eb':rel_eb,'br':float(result['br']),'psnr':float(result['psnr'])}
        points.append(point)
    points.sort(key=lambda point:point['psnr'])
    return points

def load_slice_relative_aggregate_curve(codec):
    path=project_root/'results/ocean_large_tensor_1d/comparison_figures/aggregate_metrics.csv'
    points=[]
    for result in csv.DictReader(path.open()):
        if result['compressor']==codec:points.append({'codec':codec,'rel_eb':float(result['rel_eb']),'br':float(result['br']),'psnr':float(result['normalized_psnr'])})
    points.sort(key=lambda point:point['psnr'])
    if len(points)!=len(expected_rel_ebs):raise RuntimeError(f'{codec}的Ocean-large汇总点数量错误：{len(points)}')
    return points

def comparison_record(reference,candidate):
    lower,upper,rate_difference,knots,differences=matched_quality_difference(reference,candidate)
    return {'psnr_overlap':[lower,upper],'candidate_matched_psnr_rate_difference_percent':rate_difference,'candidate_over_reference_rate_ratio':1+rate_difference/100,'comparison_knots':[{'psnr':psnr,'candidate_over_reference_rate_percent':100*(math.exp(difference)-1)}for psnr,difference in zip(knots,differences)]}

def main():
    parser=argparse.ArgumentParser(description='Summarize the five error bounds for one Ocean-large field')
    parser.add_argument('--root',type=Path,default=project_root/'results/ocean_large_field_relative_pilot')
    arguments=parser.parse_args()
    unsz=load_curve(arguments.root,'unsz')
    qoz=load_curve(arguments.root,'qoz')
    field='timeMonthly_avg_vertVelocityTop'
    field_relative=comparison_record(unsz,qoz)
    slice_relative_same_field=comparison_record(load_slice_relative_field_curve('unsz',field),load_slice_relative_field_curve('sz3',field))
    slice_relative_all_fields=comparison_record(load_slice_relative_aggregate_curve('unsz'),load_slice_relative_aggregate_curve('sz3'))
    field_excess=field_relative['candidate_matched_psnr_rate_difference_percent']
    slice_excess=slice_relative_same_field['candidate_matched_psnr_rate_difference_percent']
    rows=sorted(unsz+qoz,key=lambda point:(point['rel_eb'],point['codec']),reverse=True)
    output_path=arguments.root/'one_field_five_ebs.csv'
    with output_path.open('w',newline='')as output:
        writer=csv.DictWriter(output,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    comparison={'field':field,'interpretation':'positive rate differences mean the tensor compressor uses more bits than UNSZ at matched PSNR','field_relative_qoz_nd_vs_unsz':field_relative,'slice_relative_sz3_1d_vs_unsz_same_field':slice_relative_same_field,'slice_relative_sz3_1d_vs_unsz_all_fields':slice_relative_all_fields,'same_field_gap_change':{'rate_ratio_reduction_factor':slice_relative_same_field['candidate_over_reference_rate_ratio']/field_relative['candidate_over_reference_rate_ratio'],'excess_rate_gap_reduction_percent':100*(1-field_excess/slice_excess)}}
    (arguments.root/'one_field_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(f'per-field QoZ-ND versus UNSZ: {field_relative["candidate_matched_psnr_rate_difference_percent"]:+.3f}% bit rate at matched PSNR')
    print(f'per-slice SZ3-1D versus UNSZ on the same field: {slice_relative_same_field["candidate_matched_psnr_rate_difference_percent"]:+.3f}%')
    print(f'per-slice SZ3-1D versus UNSZ over all fields: {slice_relative_all_fields["candidate_matched_psnr_rate_difference_percent"]:+.3f}%')
    print(f'same-field tensor/UNSZ rate ratio reduction: {comparison["same_field_gap_change"]["rate_ratio_reduction_factor"]:.3f}x')
    print(f'same-field excess-rate gap reduction: {comparison["same_field_gap_change"]["excess_rate_gap_reduction_percent"]:.3f}%')
    print(f'wrote {output_path}')

if __name__=='__main__':main()
