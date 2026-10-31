import csv,json
from pathlib import Path

# 为每个消融变体生成t1任务配置：只运行UNSZ，只保留受该技术影响的field，结果写到results/ablation/<变体>/fields
project_root=Path(__file__).resolve().parents[2]
datasets=('adcirc-haiyan-1.5km','e3sm-h4','katrina','les','ocean','taps','tpost20')
excluded_fields={'RMS_T'}
csv.field_size_limit(1<<30)


def baseline_rows(task,field):
    path=project_root/task['csv_path'].format(field=field)
    return [row for row in csv.DictReader(path.open()) if row['compressor']=='unsz']


def uses_rescaling(task,field):
    return any(row['coordinate_rescaling_enabled']=='1' for row in baseline_rows(task,field))


def uses_spatial_residual(task,field):
    return any(row['selected_value_pipeline'].endswith('_spatial_residual') for row in baseline_rows(task,field))


def independent_pipeline(task):
    return {'pair':[['independent','independent']]} if 'label_config_path_list' in task['unsz'] else {'value':['independent']}


# 每个变体给出：判断field是否受影响的函数，以及要加到task.unsz中的设置
variants={
    'rescaling_off':(uses_rescaling,lambda task:{'coordinate_rescaling':'off'}),
    'tensor_prediction_off':(lambda task,field:task['slice_num']>1,lambda task:{'pipeline':independent_pipeline(task)}),
    'spatial_residual_off':(uses_spatial_residual,lambda task:{'extra_arguments':['-spatial-residual','off']}),
    'eb_schedule_off':(lambda task,field:True,lambda task:{'extra_arguments':['-eb-schedule','none']}),
    'local_ordering':(lambda task,field:True,lambda task:{'extra_arguments':['-ordering','local']}),
    'idw_weights':(lambda task,field:True,lambda task:{'extra_arguments':['-weights','idw']}),
}


def main():
    output_dir=project_root/'results/ablation/configs'
    output_dir.mkdir(parents=True,exist_ok=True)
    tasks=[task for dataset in datasets for task in json.loads((project_root/'configs'/f'{dataset}.json').read_text())]
    for name,(affected,settings) in variants.items():
        variant_tasks=[]
        for task in tasks:
            fields=[field for field in task['field_list'] if field not in excluded_fields and affected(task,field)]
            if not fields:continue
            variant_task={key:value for key,value in task.items() if key!='umc'}
            variant_task.update({'csv_path':f'results/ablation/{name}/fields/'+Path(task['csv_path']).name,'field_list':fields,'compressor_list':['unsz'],'unsz':task['unsz']|settings(task)})
            variant_tasks.append(variant_task)
        (output_dir/f'{name}.json').write_text(json.dumps(variant_tasks,indent=1,ensure_ascii=False)+'\n')
        print(f'{name}: {len(variant_tasks)} tasks, {sum(len(task["field_list"]) for task in variant_tasks)} fields')


if __name__=='__main__':
    main()
