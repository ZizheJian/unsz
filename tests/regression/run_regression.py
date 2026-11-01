import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

# 回归测试：用指定的s1_compress/s1_decompress压缩并解压一组固定用例，记录全部码流和解压结果的SHA256。
# 代码清理或增加实验开关后，正式路径的输出必须与参考二进制逐字节一致。
project_root=Path(__file__).resolve().parents[2]
data_root=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data')
ocean_large_subset=data_root/'Ocean-large/regression_subset_50k'
ocean_large_fields='{timeMonthly_avg_ssh,timeMonthly_avg_velocityMeridional,timeMonthly_avg_vertVelocityTop}'
label_config_dir=project_root/'label_configs'


def rel(eb):
    return ['-M','REL',eb]


def partitions(num):
    return ['-partition-num',str(num),'-thread-num',str(num)]


cases={
    'tpost20':[data_root/'Tpost20',['-field','scalar','-slice','[0,19]'],rel('1e-3')],
    'tpost20_1e-5':[data_root/'Tpost20',['-field','scalar','-slice','[0,19]'],rel('1e-5')],
    'tpost20_p4':[data_root/'Tpost20',['-field','scalar','-slice','[0,19]',*partitions(4)],rel('1e-3')],
    'katrina':[data_root/'Katrina',['-field','attr0','-slice','[0,161]'],['-label-config',str(label_config_dir/'Katrina_attr0_rbf_rel_1e-3.json')]],
    'katrina_p4':[data_root/'Katrina',['-field','attr0','-slice','[0,161]',*partitions(4)],['-label-config',str(label_config_dir/'Katrina_attr0_rbf_rel_1e-3.json')]],
    'e3sm_aoddust':[data_root/'E3SM-h4',['-field','AODDUST','-slice','[0,719]'],['-label-config',str(label_config_dir/'E3SM_h4_AODDUST_rbf_rel_1e-3.json')]],
    'e3sm_mass_dst_srf':[data_root/'E3SM-h4',['-field','Mass_dst_srf','-slice','[0,719]'],rel('1e-3')],
    'ocean_salinity':[data_root/'Ocean',['-field','salinity','-slice','[0,30]'],rel('1e-3')],
    'ocean_velocity_fields':[data_root/'Ocean',['-field-list','{velocityMeridional,velocityX,velocityY,velocityZ,velocityZonal,zTop}','-slice','[0,30]'],rel('1e-3')],
    'haiyan_depth':[data_root/'ADCIRC-Haiyan-1.5km',['-field','depth','-slice','[0,0]'],rel('1e-3')],
    'haiyan_zeta':[data_root/'ADCIRC-Haiyan-1.5km',['-field','zeta','-slice','[0,23]',*partitions(8)],['-label-config',str(label_config_dir/'ADCIRC_Haiyan_zeta_rbf_rel_1e-3.json')]],
    'les_velocity_x_p16':[data_root/'LES',['-field','velocity_x','-slice','[0,0]',*partitions(16)],rel('1e-3')],
    'taps_temperature_p16':[data_root/'TAPS',['-field','TEMPERATURE','-slice','[0,0]',*partitions(16)],rel('1e-3')],
    'ocean_large_slice':[ocean_large_subset,['-field-list',ocean_large_fields,'-tensor-metadata',str(ocean_large_subset/'tensor_metadata.json'),*partitions(4)],['-label-config',str(label_config_dir/'Ocean_large_monthly_rbf_rel_1e-3.json')]],
    'ocean_large_field_workers':[ocean_large_subset,['-field-list',ocean_large_fields,'-tensor-metadata',str(ocean_large_subset/'tensor_metadata.json'),*partitions(2),'-field-worker-num','2','-field-worker-map','{0,1,1}'],['-label-config',str(label_config_dir/'Ocean_large_monthly_rbf_field_rel_1e-3.json')]],
}


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb')as file:
        for chunk in iter(lambda:file.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def run_case(name,binary_dir,work_dir):
    data_dir,arguments,error_arguments=cases[name]
    case_dir=work_dir/name
    if case_dir.exists():shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    common=['-data-dir',str(data_dir),*arguments,'-coordinate-dim','3','-output-dir',str(case_dir),*error_arguments]
    result={'commands':{}}
    for stage in('compress','decompress'):
        command=[str(binary_dir/f's1_{stage}'),*common]
        result['commands'][stage]=command
        start=time.time()
        with (work_dir/f'{name}.{stage}.log').open('w')as log:
            returncode=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        result[f'{stage}_seconds']=round(time.time()-start,2)
        if returncode:
            result['error']=f'{stage} returned {returncode}'
            return result
    result['files']={path.name:sha256(path)for path in sorted(case_dir.iterdir())if path.is_file()}
    return result


def compare(manifest,reference):
    different=[]
    for name,result in manifest['cases'].items():
        expected=reference['cases'].get(name)
        if expected is None:
            print(f'{name}: 参考清单中没有该用例');different.append(name);continue
        if 'error' in result or 'error' in expected:
            print(f'{name}: 当前={result.get("error","ok")}，参考={expected.get("error","ok")}');different.append(name);continue
        if result['files']!=expected['files']:
            changed=sorted(set(result['files'])^set(expected['files'])|{file for file in set(result['files'])&set(expected['files'])if result['files'][file]!=expected['files'][file]})
            print(f'{name}: {len(changed)}个文件不同，例如{changed[:3]}');different.append(name)
        else:
            print(f'{name}: {len(result["files"])}个文件逐字节一致')
    return different


def main():
    parser=argparse.ArgumentParser(description='UNSZ逐字节回归测试')
    parser.add_argument('--binary-dir',type=Path,required=True)
    # 输出文件只用来算SHA256，默认写到scratch，不占项目目录的文件数配额
    parser.add_argument('--work-dir',type=Path,default=Path(os.environ.get('SCRATCH','/anvil/scratch/x-zjian1'))/'unsz_regression')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--compare',type=Path,help='参考清单；给出时逐用例比较SHA256')
    parser.add_argument('--case',action='append',choices=tuple(cases))
    arguments=parser.parse_args()
    binary_dir=arguments.binary_dir.resolve()
    manifest={'binary_sha256':{name:sha256(binary_dir/name)for name in('s1_compress','s1_decompress')},'cases':{}}
    for name in arguments.case or cases:
        manifest['cases'][name]=run_case(name,binary_dir,arguments.work_dir.resolve())
        result=manifest['cases'][name]
        print(f'{name}: '+(result['error'] if 'error' in result else f'{len(result["files"])} files, compress {result["compress_seconds"]} s, decompress {result["decompress_seconds"]} s'),flush=True)
    arguments.manifest.parent.mkdir(parents=True,exist_ok=True)
    arguments.manifest.write_text(json.dumps(manifest,indent=1,ensure_ascii=False))
    if arguments.compare:
        different=compare(manifest,json.loads(arguments.compare.read_text()))
        if different:sys.exit(1)


if __name__=='__main__':
    main()
