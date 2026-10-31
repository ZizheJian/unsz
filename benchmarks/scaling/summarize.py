import csv,json,os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/unsz_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root=Path(__file__).resolve().parents[2]
result_root=project_root/'results/ocean_large_scaling'
input_bytes=535724445600
field_num=95
slice_num=36264
codecs=('unsz','sz3','pfpl')
colors={'unsz':'#377eb8','sz3':'#ff7f00','pfpl':'#4daf4a'}
labels={'unsz':'UNSZ','sz3':'SZ3','pfpl':'PFPL'}

def read_rows(path):
    with path.open(newline='',encoding='utf-8') as source:
        return list(csv.DictReader(source))

def weighted_br(rows):
    return sum(float(row['br'])*int(row['input_bytes']) for row in rows)/sum(int(row['input_bytes']) for row in rows)

def validate_counts(rows,slice_key):
    actual_bytes=sum(int(row['input_bytes']) for row in rows)
    actual_slices=sum(int(row[slice_key]) for row in rows)
    if len(rows)!=field_num or actual_slices!=slice_num or actual_bytes!=input_bytes:
        raise RuntimeError(f'p1结果不完整：fields={len(rows)},slices={actual_slices},input_bytes={actual_bytes}')

def make_summary(codec,threads,rows,slice_key,max_error):
    validate_counts(rows,slice_key)
    summary={'codec':codec,'rel_eb':1e-3,'thread_num':threads,'partition_num':threads if codec=='unsz' else None,'field_num':field_num,'slice_num':slice_num,'input_bytes':input_bytes,'br':weighted_br(rows),'compression_preprocessing_time':sum(float(row['compression_preprocessing_time']) for row in rows),'compression_time':sum(float(row['compression_time']) for row in rows),'decompression_preprocessing_time':sum(float(row['decompression_preprocessing_time']) for row in rows),'decompression_time':sum(float(row['decompression_time']) for row in rows),'max_error_over_eb':max_error}
    summary['compression_total_time']=summary['compression_preprocessing_time']+summary['compression_time']
    summary['decompression_total_time']=summary['decompression_preprocessing_time']+summary['decompression_time']
    return summary

def prepare_p1():
    unsz_rows=[]
    for path in sorted((project_root/'results/ocean_large_tensor_full/sparse_window').glob('*.csv')):
        unsz_rows.extend(read_rows(path))
    summaries={'unsz':make_summary('unsz',1,unsz_rows,'slice_num',None)}
    summaries['unsz']['source']='results/ocean_large_tensor_full/sparse_window'
    for codec in ('sz3','pfpl'):
        source=project_root/f'results/ocean_large_tensor_1d/{codec}/rel_0.001_t1/summary.csv'
        rows=read_rows(source)
        if not all(row['valid'].lower()=='true' for row in rows):
            raise RuntimeError(f'{codec} p1包含无效结果')
        summary=make_summary(codec,1,rows,'slice_num',max(float(row['max_error_over_eb']) for row in rows))
        summary['source']=str(source.relative_to(project_root))
        summaries[codec]=summary
    for codec,summary in summaries.items():
        suffix='p1' if codec=='unsz' else 't1'
        output_dir=result_root/codec/f'rel_0.001_{suffix}'
        output_dir.mkdir(parents=True,exist_ok=True)
        (output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

def load_summaries():
    points=[]
    for codec in codecs:
        suffix='p' if codec=='unsz' else 't'
        for path in sorted((result_root/codec).glob(f'rel_0.001_{suffix}*/summary.json')):
            summary=json.loads(path.read_text())
            if summary.get('codec')!=codec or summary.get('field_num')!=field_num or summary.get('slice_num')!=slice_num or summary.get('input_bytes')!=input_bytes:
                raise RuntimeError(f'scaling summary不完整：{path}')
            points.append(summary)
    points.sort(key=lambda point:(codecs.index(point['codec']),point['thread_num']))
    return points

def write_table(points):
    references={point['codec']:point for point in points if point['thread_num']==1}
    rows=[]
    for point in points:
        reference=references[point['codec']]
        row=dict(point)
        row['compression_speedup']=reference['compression_total_time']/point['compression_total_time']
        row['decompression_speedup']=reference['decompression_total_time']/point['decompression_total_time']
        row['compression_efficiency']=row['compression_speedup']/point['thread_num']
        row['decompression_efficiency']=row['decompression_speedup']/point['thread_num']
        rows.append(row)
    keys=('codec','thread_num','partition_num','br','compression_preprocessing_time','compression_time','compression_total_time','compression_speedup','compression_efficiency','decompression_preprocessing_time','decompression_time','decompression_total_time','decompression_speedup','decompression_efficiency','max_error_over_eb','field_num','slice_num','input_bytes')
    with (result_root/'summary.csv').open('w',newline='') as destination:
        writer=csv.DictWriter(destination,fieldnames=keys,lineterminator='\n')
        writer.writeheader()
        writer.writerows({key:row.get(key) for key in keys} for row in rows)
    return rows

def plot(rows):
    figure,axes=plt.subplots(1,3,figsize=(10.8,3.6))
    max_thread=max(row['thread_num'] for row in rows)
    for codec in codecs:
        codec_rows=[row for row in rows if row['codec']==codec]
        if not codec_rows:
            continue
        threads=[row['thread_num'] for row in codec_rows]
        axes[0].plot(threads,[row['compression_speedup'] for row in codec_rows],marker='o',color=colors[codec],label=labels[codec])
        axes[1].plot(threads,[row['decompression_speedup'] for row in codec_rows],marker='o',color=colors[codec],label=labels[codec])
        axes[2].plot(threads,[row['br'] for row in codec_rows],marker='o',color=colors[codec],label=labels[codec])
    ideal_threads=[1,2,4,8,16,32]
    ideal_threads=[thread for thread in ideal_threads if thread<=max_thread]
    for axis in axes[:2]:
        axis.plot(ideal_threads,ideal_threads,color='#777777',linestyle='--',linewidth=1,label='Ideal')
        axis.set_ylabel('Speedup')
        axis.set_ylim(bottom=0)
    axes[2].set_ylabel('Bit rate (bits/value)')
    axes[2].set_yscale('log')
    for axis in axes:
        axis.set_xscale('log',base=2)
        axis.set_xticks(ideal_threads)
        axis.set_xticklabels([str(thread) for thread in ideal_threads])
        axis.set_xlabel('Threads / partitions')
        axis.grid(True,alpha=0.25)
    axes[0].text(0.5,-0.43,'(a) Compression-side scaling',transform=axes[0].transAxes,ha='center',va='top',fontsize=11)
    axes[1].text(0.5,-0.43,'(b) Decompression-side scaling',transform=axes[1].transAxes,ha='center',va='top',fontsize=11)
    axes[2].text(0.5,-0.43,'(c) Bit-rate stability',transform=axes[2].transAxes,ha='center',va='top',fontsize=11)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,legend_labels,loc='upper center',ncol=len(legend_labels),frameon=False,bbox_to_anchor=(0.5,1.03))
    figure.subplots_adjust(left=0.065,right=0.99,top=0.82,bottom=0.3,wspace=0.3)
    figure_dir=result_root/'figures'
    figure_dir.mkdir(parents=True,exist_ok=True)
    figure.savefig(figure_dir/'scaling.pdf',bbox_inches='tight')
    figure.savefig(figure_dir/'scaling.png',dpi=220,bbox_inches='tight')
    plt.close(figure)

def main():
    prepare_p1()
    rows=write_table(load_summaries())
    plot(rows)
    print(f'Wrote {len(rows)} scaling points to {result_root}')

if __name__=='__main__':
    main()
