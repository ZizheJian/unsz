import argparse,csv,hashlib,json,math,textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

project_root=Path(__file__).resolve().parents[2]
codecs=('sz3','pfpl')
rel_ebs=(1e-5,1e-4,1e-3,1e-2,1e-1)
styles={'unsz':('UNSZ','#CC3311','D'),'umc':('UMC','#AA4499','P'),'sz3':('SZ3','#0072B2','o'),'pfpl':('PFPL','#009E73','^')}


def read_results(root):
    records={}
    sources=[]
    codec_hashes={}
    metadata_hashes=set()
    for codec in codecs:
        for eb in rel_ebs:
            directory=root/codec/f'rel_{eb:g}_t1'
            manifest=json.loads((directory/'manifest.json').read_text())
            if manifest['codec']!=codec or manifest['threads']!=1 or float(manifest['rel_eb'])!=eb or manifest['codec_error_bound_mode']!='ABS':
                raise ValueError(f'Baseline identity mismatch: {directory}')
            codec_hashes.setdefault(codec,set()).add(manifest['binary_sha256'])
            metadata_hashes.add(manifest['metadata_sha256'])
            path=directory/'summary.csv'
            with path.open(newline='') as source:
                rows=list(csv.DictReader(source))
            if len(rows)!=95:
                raise ValueError(f'Expected 95 fields: {path},got {len(rows)}')
            for row in rows:
                key=(row['field'],codec,eb)
                if key in records or row['compressor']!=codec or int(row['thread_num'])!=1 or float(row['rel_eb'])!=eb:
                    raise ValueError(f'Duplicate or mismatched record: {key}')
                row=dict(row)
                for name in ('slice_num','completed_slice_num','input_bytes','compressed_bytes','zero_eb_slice_num','returncode'):
                    row[name]=int(row[name])
                for name in ('br','psnr','compression_time','decompression_time','compression_preprocessing_time','decompression_preprocessing_time','max_error_over_eb'):
                    row[name]=float(row[name])
                row['valid']=row['valid']=='True'
                if row['slice_num']!=row['completed_slice_num'] or not math.isclose(row['br'],32*row['compressed_bytes']/row['input_bytes'],rel_tol=1e-12):
                    raise ValueError(f'Incomplete slices or inconsistent bit rate: {key}')
                if row['returncode']!=0 or not row['valid']:
                    raise ValueError(f'Unexpected codec failure: {key}')
                if row['valid']!=(row['returncode']==0 and row['max_error_over_eb']<=1.01):
                    raise ValueError(f'Inconsistent error status: {key}')
                if row['input_bytes']<=0 or not math.isfinite(row['br']) or row['br']<=0 or math.isnan(row['psnr']) or row['psnr']==-math.inf:
                    raise ValueError(f'Invalid numerical metric: {key}')
                for operation in ('compression','decompression'):
                    if row[operation+'_time']<=0:
                        raise ValueError(f'Nonpositive timing: {key}')
                    row[operation+'_MB_s']=row['input_bytes']/1e6/row[operation+'_time']
                records[key]=row
            if sum(row['completed_slice_num'] for row in records.values() if row['compressor']==codec and float(row['rel_eb'])==eb)!=36264:
                raise ValueError(f'Expected all 36264 slices: {directory}')
            sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'manifest':manifest})
    if len(metadata_hashes)!=1 or any(len(hashes)!=1 for hashes in codec_hashes.values()):
        raise ValueError('Mixed metadata or mixed binaries within a codec')
    names={key[0] for key in records}
    for field in names:
        rows=[records[field,codec,eb] for codec in codecs for eb in rel_ebs]
        if len({row['input_bytes'] for row in rows})!=1 or len({row['slice_num'] for row in rows})!=1:
            raise ValueError(f'Input sizes differ: {field}')
    fields=sorted(names,key=lambda field:(records[field,'sz3',1e-3]['input_bytes'],field.casefold()))
    return records,fields,sources


def caption(field):
    name=field.removeprefix('timeMonthly_avg_')
    parts=name.split('_')
    return '\n'.join(line for part in ['Ocean-large - '+parts[0],*parts[1:]] for line in textwrap.wrap(part,width=56,break_long_words=True,break_on_hyphens=False))


def grid(count):
    columns=min(6,count)
    figure,axes=plt.subplots(math.ceil(count/columns),columns,figsize=(3.8*columns,3.5*math.ceil(count/columns)),squeeze=False,layout='constrained')
    axes=axes.ravel()
    for axis in axes[count:]:
        axis.remove()
    return figure,axes[:count]


def finish(figure,handles,name,output,description,legend_columns=4):
    figure.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.002),ncol=legend_columns,frameon=False)
    figure.text(.5,-.012,description,ha='center',va='top',fontsize=11)
    for suffix in ('png','pdf'):
        figure.savefig(output/f'{name}.{suffix}',dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(figure)


def ordered_codecs(records):
    present={row['compressor'] for row in records.values()}
    return tuple(codec for codec in styles if codec in present)


def codec_handles(selected=codecs,records=None):
    handles=[]
    for codec in selected:
        errors=sorted({float(row['rel_eb']) for row in records.values() if row['compressor']==codec}) if records is not None else list(rel_ebs)
        suffix=f' ({len(errors)}/{len(rel_ebs)} EB)' if len(errors)<len(rel_ebs) else ''
        handles.append(Line2D([],[],label=styles[codec][0]+suffix,color=styles[codec][1],marker=styles[codec][2],linewidth=1.5,linestyle='none' if len(errors)==1 else '-'))
    return handles


def structured_result_directories(fields,codec,directory):
    complete=[]
    skipped=[]
    prefix='rel_' if codec=='unsz' else 'serial_rel_'
    for eb,tag in ((1e-5,'1e-5'),(1e-4,'1e-4'),(1e-3,'1e-3'),(1e-2,'1e-2'),(1e-1,'1e-1')):
        path=directory if eb==1e-3 else directory.parent/(prefix+tag)
        present={item.stem for item in path.glob('*.csv')}
        if present-set(fields):
            raise ValueError(f'Unexpected structured-codec fields: {path}')
        if present!=set(fields):
            skipped.append({'rel_eb':eb,'path':str(path.resolve()),'completed_field_num':len(present),'expected_field_num':len(fields)})
        else:
            complete.append((eb,path))
    return complete,skipped


def add_structured_results(records,fields,sources,codec,directory,expected_eb=1e-3):
    manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if float(manifest['rel_eb'])!=expected_eb or expected_eb not in rel_ebs or manifest['field_num']!=95 or manifest.get('thread_num',manifest.get('program_threads'))!=1 or manifest.get('partition_num',1)!=1:
        raise ValueError(f'Unexpected structured-codec configuration: {directory}')
    if manifest.get('metadata_sha256',sources[0]['manifest']['metadata_sha256'])!=sources[0]['manifest']['metadata_sha256']:
        raise ValueError(f'Structured-codec metadata differs: {directory}')
    if any(item.get('codec')==codec and item['manifest']['binary_sha256']!=manifest['binary_sha256'] for item in sources):
        raise ValueError(f'Mixed structured-codec binaries: {codec}')
    keys=list(next(iter(records.values())))
    for field in fields:
        path=directory/(field+'.csv')
        with path.open(newline='') as source:
            rows=[row for row in csv.DictReader(source) if row['compressor']==codec and row['field']==field]
        if len(rows)!=1:
            raise ValueError(f'Expected one current structured-codec row: {path}')
        source=rows[0]
        eb=float(source['rel_eb']) if source['rel_eb']!='not_applicable' else None
        label_path=None
        if eb is None:
            label_path=Path(source['label_config'])
            if not label_path.is_absolute():
                label_path=project_root/label_path
            labels=json.loads(label_path.read_text())['labels']
            errors={float(label['eb_value']) for label in labels if label['method']=='predict' and label['eb_mode']=='REL'}
            if len(errors)!=1:
                raise ValueError(f'Expected one relative error bound: {label_path}')
            eb=errors.pop()
        reference=records[field,'sz3',1e-3]
        if eb!=expected_eb or int(source['input_bytes'])!=reference['input_bytes'] or int(source['slice_num'])!=reference['slice_num']:
            raise ValueError(f'Structured-codec inputs differ: {path}')
        if (field,codec,eb) in records:
            raise ValueError(f'Duplicate structured-codec result: {field},{codec},{eb}')
        row={key:'' for key in keys}
        row.update({'field':field,'compressor':codec,'rel_eb':eb,'thread_num':1,'slice_num':reference['slice_num'],'completed_slice_num':reference['slice_num'],'input_bytes':reference['input_bytes'],'br':float(source['br']),'psnr':float(source['psnr']),'valid':True,'returncode':0,'max_error_over_eb':math.nan})
        # UNSZ prints BR to six decimals; do not invent byte-exact payload sizes.
        row['compressed_bytes']=row['br']*row['input_bytes']/32
        for name in ('compression_time','decompression_time','compression_preprocessing_time','decompression_preprocessing_time'):
            row[name]=float(source[name])
            if not math.isfinite(row[name]) or row[name]<0:
                raise ValueError(f'Invalid structured-codec timing: {path}')
        if row['br']<=0 or not math.isfinite(row['br']) or math.isnan(row['psnr']) or row['psnr']==-math.inf:
            raise ValueError(f'Invalid structured-codec RD metrics: {path}')
        for operation in ('compression','decompression'):
            if row[operation+'_time']<=0:
                raise ValueError(f'Nonpositive structured-codec timing: {path}')
            row[operation+'_MB_s']=row['input_bytes']/1e6/row[operation+'_time']
        records[field,codec,eb]=row
        sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'codec':codec,'manifest':manifest,'validation':'UNSZ independent decompression checked by its runner; UMC additionally checked against on-disk reconstruction','compressed_bytes_rule':'Derived from recorded field BR; UNSZ BR has six-decimal precision','label_config_sha256':hashlib.sha256(label_path.read_bytes()).hexdigest() if label_path else None})
    sources.append({'path':str(manifest_path.resolve()),'sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'codec':codec,'manifest':manifest})


def aggregate_rows(rows,codec,eb):
    item={'compressor':codec,'rel_eb':eb,'field_num':len(rows),'slice_num':sum(row['slice_num'] for row in rows),'input_bytes':sum(row['input_bytes'] for row in rows),'compressed_bytes':sum(row['compressed_bytes'] for row in rows)}
    for name in ('compression_time','decompression_time','compression_preprocessing_time','decompression_preprocessing_time'):
        item[name]=sum(row[name] for row in rows)
    item['br']=32*item['compressed_bytes']/item['input_bytes']
    item['nmse']=math.fsum(row['input_bytes']*10**(-row['psnr']/10) for row in rows)/item['input_bytes']
    item['normalized_psnr']=math.inf if item['nmse']==0 else -10*math.log10(item['nmse'])
    item['compression_MB_s']=item['input_bytes']/1e6/item['compression_time']
    item['decompression_MB_s']=item['input_bytes']/1e6/item['decompression_time']
    item['invalid_field_num']=sum(not row['valid'] for row in rows)
    item['max_error_over_eb']=max(row['max_error_over_eb'] for row in rows)
    item['valid']=item['invalid_field_num']==0
    return item


def field_groups(metadata_path,fields,sources,records):
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest()!=sources[0]['manifest']['metadata_sha256']:
        raise ValueError('Tensor metadata differs from the benchmark input')
    metadata=json.loads(metadata_path.read_text())['fields']
    if set(metadata)!=set(fields):
        raise ValueError('Field set differs from tensor metadata')
    groups={shape:[] for shape in ('12x1','12x80','12x81')}
    for field in fields:
        shape='x'.join(str(value) for value in metadata[field]['shape'])
        if shape not in groups or math.prod(metadata[field]['shape'])!=records[field,'sz3',1e-3]['slice_num']:
            raise ValueError(f'Unexpected tensor shape: {field}')
        groups[shape].append(field)
    if [len(group) for group in groups.values()]!=[58,33,4]:
        raise ValueError('Expected shape groups containing 58,33,4 fields')
    for group in groups.values():
        if len({records[field,'sz3',1e-3]['input_bytes'] for field in group})!=1:
            raise ValueError('Fields within a shape group must have equal point counts')
    return groups


def draw_group(axis,grouped,shape,throughput):
    for codec in ordered_codecs(grouped):
        _,color,marker=styles[codec]
        curve=[grouped[shape,codec,eb] for eb in rel_ebs if (shape,codec,eb) in grouped]
        if not throughput:
            curve=sorted(curve,key=lambda row:row['br'])
        operations=(('compression_MB_s','-'),('decompression_MB_s','--')) if throughput else (('normalized_psnr','-'),)
        for metric,linestyle in operations:
            x=[row['rel_eb'] if throughput else row['br'] for row in curve]
            axis.plot(x,[row[metric] for row in curve],color=color,marker=marker,linewidth=1.7,markersize=7 if len(curve)==1 else 4,linestyle=linestyle if len(curve)>1 else 'none',markerfacecolor='none' if metric=='decompression_MB_s' else color,zorder=5 if len(curve)==1 else 2)
    axis.set_xscale('log')
    if throughput:
        axis.set_ylim(bottom=0)
    axis.grid(True,which='major',linestyle=':',alpha=.5)
    axis.set_xlabel(f"Ocean-large - {shape.replace('x','×')} ({curve[0]['field_num']} fields)",labelpad=9)


def plot_grouped(grouped,groups,output,throughput=False):
    handles=codec_handles(ordered_codecs(grouped),grouped)
    if throughput:
        handles.extend(Line2D([],[],color='.3',linestyle=linestyle,label=label) for label,linestyle in (('Compression','-'),('Decompression','--')))
    name='releb_throughput' if throughput else 'rd'
    xlabel='REL EB (log scale)' if throughput else 'Bit rate (bit/value, log scale)'
    ylabel='Core throughput (MB/s)' if throughput else 'Aggregated normalized PSNR (dB)'
    description='Single thread; all fields included; timings from separate jobs.'
    figure,axes=plt.subplots(1,3,figsize=(13.5,4),layout='constrained')
    for shape,axis in zip(groups,axes):
        draw_group(axis,grouped,shape,throughput)
    figure.supxlabel(xlabel)
    figure.supylabel(ylabel)
    finish(figure,handles,'grouped_'+name,output,description,4 if throughput else 3)
    for shape in groups:
        figure,axis=plt.subplots(figsize=(5.2,4),layout='constrained')
        draw_group(axis,grouped,shape,throughput)
        figure.supxlabel(xlabel)
        figure.supylabel(ylabel)
        finish(figure,handles,'shape_'+shape+'_'+name,output,description,2)


def plot_rd(records,fields,output):
    figure,axes=grid(len(fields))
    for field,axis in zip(fields,axes):
        exact=[]
        finite_num=0
        for codec in ordered_codecs(records):
            label,color,marker=styles[codec]
            curve=sorted((records[field,codec,eb] for eb in rel_ebs if (field,codec,eb) in records),key=lambda row:row['br'])
            finite=[row for row in curve if math.isfinite(row['psnr'])]
            finite_num+=len(finite)
            axis.plot([row['br'] for row in finite],[row['psnr'] for row in finite],color=color,marker=marker,markersize=3,linewidth=1.3)
            infinite=[row for row in curve if row['psnr']==math.inf]
            if infinite:
                rates=sorted({row['br'] for row in infinite})
                rate_label=f'{rates[0]:.4g}' if len(rates)==1 else f'{rates[0]:.4g}–{rates[-1]:.4g}'
                exact.append(f'{label}: {rate_label} bit/value')
        if exact:
            axis.text(.5,.5 if not finite_num else .98,'Exact reconstruction (PSNR=∞)\n'+'\n'.join(exact),transform=axis.transAxes,ha='center',va='center' if not finite_num else 'top',fontsize=7)
        axis.set_xscale('log')
        if not finite_num:
            rates=[records[field,codec,eb]['br'] for codec in codecs for eb in rel_ebs]
            axis.set_xlim(min(rates)*.8,max(rates)*1.2)
            axis.set_yticks([])
        axis.set_xlabel(caption(field),fontsize=7,labelpad=7)
        axis.grid(True,linestyle=':',alpha=.45)
    figure.supxlabel('Bit rate (bit/value, log scale)')
    figure.supylabel('PSNR (dB)')
    handles=codec_handles(ordered_codecs(records),records)
    finish(figure,handles,'all_fields_rd',output,'Ocean-large: measured rate-distortion.')


def plot_throughput(records,fields,output):
    figure,axes=grid(len(fields))
    for field,axis in zip(fields,axes):
        for codec in ordered_codecs(records):
            _,color,marker=styles[codec]
            curve=[records[field,codec,eb] for eb in rel_ebs if (field,codec,eb) in records]
            for operation,linestyle in (('compression','-'),('decompression','--')):
                axis.plot([row['rel_eb'] for row in curve],[row[operation+'_MB_s'] for row in curve],color=color,marker=marker,markersize=3,linewidth=1.3,linestyle=linestyle)
        axis.set_xscale('log')
        axis.set_ylim(bottom=0)
        axis.set_xlabel(caption(field),fontsize=7,labelpad=7)
        axis.grid(True,which='both',linestyle=':',alpha=.45)
    figure.supxlabel('REL EB (log scale; each slice encoded with its corresponding ABS EB)')
    figure.supylabel('Core throughput (MB/s)')
    handles=codec_handles(ordered_codecs(records),records)+[Line2D([],[],color='.3',linestyle=linestyle,label=label) for label,linestyle in (('Compression','-'),('Decompression','--'))]
    handles.append(Line2D([],[],color='black',marker='x',linestyle='none',label='ABS error bound exceeded'))
    finish(figure,handles,'all_fields_releb_throughput',output,'Ocean-large: single-thread core throughput; preprocessing excluded.',3)


def time_handles():
    return [Patch(facecolor='.6',hatch='..',label='Compression preprocessing'),Patch(facecolor='.6',label='Compression'),Patch(facecolor='.6',hatch='\\\\',label='Decompression preprocessing'),Patch(facecolor='.6',hatch='///',label='Decompression'),Patch(facecolor='white',hatch='xx',edgecolor='black',label='ABS error bound exceeded')]


def draw_time(axis,records,fields,aggregate=False):
    selected=ordered_codecs(records)
    width=.8/len(selected)
    for index,field in enumerate(fields):
        for codec_index,codec in enumerate(selected):
            row=records[codec] if aggregate else records[field,codec,1e-3]
            _,color,_=styles[codec]
            x=index+(codec_index-(len(selected)-1)/2)*width
            bottom=0
            for name,hatch in (('compression_preprocessing_time','..'),('compression_time',''),('decompression_preprocessing_time','\\\\'),('decompression_time','///')):
                axis.bar(x,row[name],bottom=bottom,width=width*.9,color=color,edgecolor='black',linewidth=.3,hatch=hatch)
                bottom+=row[name]
            if not row['valid']:
                axis.bar(x,bottom,width=width*.9,facecolor='none',edgecolor='black',linewidth=.7,hatch='xx')
            if not aggregate:
                axis.text(x,0,styles[codec][0],transform=axis.get_xaxis_transform(),rotation=90,ha='center',va='bottom',fontsize=6)
    if aggregate:
        axis.set_xticks([(i-(len(selected)-1)/2)*width for i in range(len(selected))],[styles[codec][0] for codec in selected])
    else:
        axis.set_xticks(range(len(fields)),[caption(field) for field in fields],rotation=28,ha='right',fontsize=7)
    axis.set_ylim(bottom=0)
    axis.grid(True,axis='y',linestyle=':',alpha=.45)


def plot_time(records,fields,aggregate,output,per_field=False):
    handles=[Patch(facecolor=styles[codec][1],label=styles[codec][0]) for codec in ordered_codecs(records)]+time_handles()
    if per_field:
        groups=[fields[begin:begin+3] for begin in range(0,len(fields),3)]
        figure,axes=grid(len(groups))
        for group,axis in zip(groups,axes):
            draw_time(axis,records,group)
        figure.supylabel('Time (s)')
        finish(figure,handles,'all_fields_releb_1e-3_time',output,'Ocean-large: REL=1e-3, one thread; both preprocessing stages included; all fields included.',3)
    figure,axis=plt.subplots(figsize=(10,4.5),layout='constrained')
    draw_time(axis,{codec:aggregate[codec,1e-3] for codec in ordered_codecs(aggregate)},['all'],aggregate=True)
    axis.set_ylabel('Time (s)')
    finish(figure,handles,'total_releb_1e-3_time',output,'Ocean-large: REL=1e-3 total time over all 95 fields,including both preprocessing stages.',5)


def write_csv(path,rows):
    with path.open('w',newline='') as destination:
        writer=csv.DictWriter(destination,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description='Plot complete Ocean-large 1D baselines; ZFP is excluded because it violates the ABS bound on fields with sentinel values')
    parser.add_argument('--results-root',type=Path,default=project_root/'results/ocean_large_tensor_1d')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--metadata',type=Path,default=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data/tensor_metadata.json'))
    parser.add_argument('--per-field',action='store_true',help='Also regenerate the complete per-field figures')
    parser.add_argument('--include-structured',action='store_true',help='Overlay all complete single-thread UNSZ and UMC EB results')
    parser.add_argument('--unsz-results',type=Path,default=project_root/'results/ocean_large_tensor_full/sparse_window')
    parser.add_argument('--umc-results',type=Path,default=project_root/'results/ocean_large_tensor_full_umc/serial_rel_1e-3')
    arguments=parser.parse_args()
    output=arguments.output_dir or arguments.results_root/('comparison_figures' if arguments.include_structured else 'figures')
    records,fields,sources=read_results(arguments.results_root)
    groups=field_groups(arguments.metadata,fields,sources,records)
    skipped={}
    if arguments.include_structured:
        for codec,directory in (('unsz',arguments.unsz_results),('umc',arguments.umc_results)):
            complete,skipped[codec]=structured_result_directories(fields,codec,directory)
            if not any(eb==1e-3 for eb,_ in complete):
                raise ValueError(f'Expected complete REL=1e-3 structured-codec result: {directory}')
            for eb,path in complete:
                add_structured_results(records,fields,sources,codec,path,eb)
            for item in skipped[codec]:
                print(f'Skip incomplete {codec} REL={item["rel_eb"]:g}: {item["completed_field_num"]}/{item["expected_field_num"]} fields')
    selected=ordered_codecs(records)
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.formatter.useoffset':False})
    aggregate={}
    for codec in selected:
        for eb in rel_ebs:
            rows=[records[field,codec,eb] for field in fields if (field,codec,eb) in records]
            if rows:
                aggregate[codec,eb]=aggregate_rows(rows,codec,eb)
    grouped={(shape,codec,eb):aggregate_rows([records[field,codec,eb] for field in group],codec,eb) for shape,group in groups.items() for codec in selected for eb in rel_ebs if (codec,eb) in aggregate}
    write_csv(output/'all_fields_metrics.csv',[records[field,codec,eb] for field in fields for codec in selected for eb in rel_ebs if (field,codec,eb) in records])
    write_csv(output/'aggregate_metrics.csv',list(aggregate.values()))
    write_csv(output/'grouped_metrics.csv',[{'shape':shape,**row} for (shape,_,_),row in grouped.items()])
    coverage={codec:[eb for eb in rel_ebs if (codec,eb) in aggregate] for codec in selected}
    (output/'manifest.json').write_text(json.dumps({'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sources':sources,'field_order':fields,'shape_groups':groups,'regenerated_per_field':arguments.per_field,'throughput_rule':'sum(input bytes)/sum(core time)/1e6,not mean field throughput','rd_rule':'BR=sum(compressed bytes)*32/sum(input bytes); NMSE=point-weighted mean of 10**(-field PSNR/10); grouped PSNR=-10*log10(NMSE); infinite field PSNR contributes zero NMSE; equal-sized fields within each shape imply arithmetic means','invalid_rule':'Every plotted result satisfies its ABS bound; ZFP is excluded because it violates the bound on fields with sentinel values','preprocessing_rule':'Both preprocessing stages included in total time; zero for 1D codecs; UNSZ shares untransformed spatial DAG across fields; current UMC serial runner preprocesses each field separately','timing_comparison':'Separate jobs,nodes are not paired; not a controlled same-node speedup experiment','eb_coverage':coverage,'skipped_incomplete_results':skipped,'structured_coverage':'Only EB runs containing all 95 fields included; raw input and slice counts match every baseline field; recorded BR rounded for UNSZ'},indent=2)+'\n')
    plot_grouped(grouped,groups,output)
    plot_grouped(grouped,groups,output,throughput=True)
    if arguments.per_field:
        plot_rd(records,fields,output)
        plot_throughput(records,fields,output)
    plot_time(records,fields,aggregate,output,arguments.per_field)
    print(output.resolve())
    for codec in selected:
        print(json.dumps(aggregate[codec,1e-3]))


if __name__=='__main__':
    main()
