import contextlib,csv,hashlib,io,json,math,os,socket,subprocess,sys,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project_root))
data_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")
result_root=project_root/"results/current"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
temporal_configuration_set={"constant_extrapolation","linear_interpolation","constant_extrapolation_spatial_residual","linear_interpolation_spatial_residual"}

def field_dag_cache_path(field):
    return data_root/field["folder"]/".unsz_cache"/"benchmarks"/f"{field['key']}.dag"

def select_benchmark_driver(field):
    configuration=field["config"]
    if configuration in temporal_configuration_set:
        return "temporal"
    if configuration.startswith("reverse_") or configuration.startswith("forward_"):
        return "geometric_order"
    if configuration.startswith("idw_") or configuration.startswith("rbf_"):
        return "reference_order"
    raise ValueError(f"{field['key']}具有未知configuration：{configuration}")

def run_command(command):
    completed=subprocess.run([str(value) for value in command],cwd=project_root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(completed.stdout,end="",flush=True)
    if completed.returncode!=0:
        raise RuntimeError("命令执行失败："+" ".join(str(value) for value in command))
    return completed.stdout

def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))

def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

def build_unsz_command(field,rel_eb,result_path,result_root_path):
    data_dir=data_root/field["folder"]
    benchmark_driver=select_benchmark_driver(field)
    if benchmark_driver=="temporal":
        command=[sys.executable,project_root/"py_codes/benchmark_field_temporal.py","-data-dir",data_dir,"-variable",field["variable"],"-time-step-num",field["steps"],"-dim",field["dim"],"-configuration",field["config"],"-predictor",field.get("predictor","inverse_proportion"),"-ref-num",field["ref_num"],"-rel-eb",rel_eb,"-result",result_path]
        if field.get("predictor")=="rbf_fitting":
            command.extend(("-rbf-shape",field["shape"]))
        if "matrix" in field:
            command.extend(("-matrix-file",project_root/"adjust_coord_matrices"/field["matrix"]))
        if "label_pattern" in field:
            command.extend(("-label-file-pattern",str(data_dir/field["label_pattern"])))
        if "condition" in field:
            command.extend(("-condition-operator",field["condition"][0],"-condition-value",field["condition"][1]))
        if "time_step_coordinate_file" in field:
            command.extend(("-time-step-coordinate-file",data_dir/field["time_step_coordinate_file"]))
        return command
    if benchmark_driver=="geometric_order":
        command=[sys.executable,project_root/"py_codes/screen_field_geometric_order.py","-data-dir",data_dir,"-variable",field["variable"],"-dim",field["dim"],"-configuration",field["config"],"-ref-num",field["ref_num"],"-rbf-shape",field["shape"],"-rel-eb",rel_eb,"-result",result_path]
        return command
    prediction_ref_num=field.get("prediction_ref_num",field.get("ref_num"))
    if prediction_ref_num is None:
        raise ValueError(f"{field['key']}缺少prediction_ref_num")
    command=["-data-dir",data_dir,"-variable",field["variable"],"-time-step-num",field["steps"],"-dim",field["dim"],"-configuration",field["config"],"-rel-eb",rel_eb,"-result",result_path]
    if field["config"].startswith("idw_"):
        command.extend(("-idw-ref-num",prediction_ref_num,"-idw-power",field.get("power",2)))
    else:
        command.extend(("-rbf-ref-num",prediction_ref_num,"-rbf-shape",field["shape"]))
        if "ordering_score_min_ref_num" in field:
            command.extend(("-ordering-score-max-ref-num",field["ordering_score_max_ref_num"],"-ordering-score-min-ref-num",field["ordering_score_min_ref_num"]))
    if "ordering_candidate_num" in field:
        command.extend(("-order-candidate-num",field["ordering_candidate_num"]))
    if "matrix" in field:
        command.extend(("-matrix-file",project_root/"adjust_coord_matrices"/field["matrix"]))
    for time_step in field.get("training",[]):
        command.extend(("-training-time-step",time_step))
    if "condition" in field:
        command.extend(("-condition-operator",field["condition"][0],"-condition-value",field["condition"][1]))
    if field.get("reorder_by_prediction_order"):
        command.append("-reorder-data-by-prediction-order")
    if field.get("disable_eb_schedule"):
        command.append("-disable-eb-schedule")
    for option in field.get("common_extra",[]):
        command.append(f"-common-extra={option}")
    for option in field.get("compression_extra",[]):
        command.append(f"-compression-extra={option}")
    for option in field.get("decompression_extra",[]):
        command.append(f"-decompression-extra={option}")
    if field.get("persist_dag"):
        command.extend(("-dag-cache-result",field_dag_cache_path(field)))
    return command

def run_unsz(field,rel_eb,result_root_path):
    temporary_result=result_root_path/"partial"/f"{field['key']}_unsz_{rel_eb:g}.csv"
    command=build_unsz_command(field,rel_eb,temporary_result,result_root_path)
    if select_benchmark_driver(field)=="reference_order":
        import t1_start_c_code as benchmark
        benchmark.reference_order_main([str(value) for value in command])
        command_description=["t1_start_c_code.reference_order_main",*command]
    else:
        run_command(command)
        command_description=command
    source=read_rows(temporary_result)
    if len(source)!=1:
        raise RuntimeError(f"{temporary_result}应恰好包含一行，实际为{len(source)}")
    row=source[0]
    return {"compressor":"unsz","rel_eb":rel_eb,"br":float(row["br"]),"psnr":float(row["psnr"]),"preprocessing_time":float(row["preprocessing_time"]),"compression_time":float(row["compression_time"]),"decompression_time":float(row["decompression_time"]),"command":" ".join(str(value) for value in command_description)}

def run_umc(field,rel_eb):
    import t1_start_c_code as benchmark
    data_dir=data_root/field["folder"]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    output=io.StringIO()
    with tempfile.TemporaryDirectory(prefix=f"umc_{field['key']}_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        (temporary_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        (temporary_dir/"conn.dat").symlink_to(data_dir/"conn.dat")
        for time_step in range(field["steps"]):
            (temporary_dir/f"{field['variable']}.dat.{time_step}").symlink_to(data_dir/f"{field['variable']}.dat.{time_step}")
        dataset={"name":field["key"],"data_dir":str(temporary_dir),"folder":field["folder"],"variable":field["variable"],"time_step_num":field["steps"],"dim":field["dim"],**field["umc"]}
        try:
            with contextlib.redirect_stdout(output):
                br,psnr,preprocessing_throughput,compression_throughput,decompression_throughput=benchmark.test("umc",dataset,rel_eb)
        except Exception:
            print(output.getvalue(),end="",flush=True)
            raise
    print(output.getvalue(),end="",flush=True)
    input_mb=sum((data_dir/f"{field['variable']}.dat.{time_step}").stat().st_size for time_step in range(field["steps"]))/1e6
    return {"compressor":"umc","rel_eb":rel_eb,"br":br,"psnr":psnr,"preprocessing_time":input_mb/preprocessing_throughput,"compression_time":input_mb/compression_throughput,"decompression_time":input_mb/decompression_throughput,"command":"t1_start_c_code.test(umc)"}

def write_manifest(field,result_root_path,rows):
    data_dir=data_root/field["folder"]
    input_paths=[data_dir/"coords.dat",*[data_dir/f"{field['variable']}.dat.{time_step}" for time_step in range(field["steps"])]]
    if field.get("umc"):
        input_paths.append(data_dir/"conn.dat")
    if "label_pattern" in field:
        input_paths.extend(data_dir/field["label_pattern"].format(time_step=time_step) for time_step in range(field["steps"]))
    if "time_step_coordinate_file" in field:
        input_paths.append(data_dir/field["time_step_coordinate_file"])
    if "matrix" in field:
        input_paths.append(project_root/"adjust_coord_matrices"/field["matrix"])
    dag_path=field_dag_cache_path(field)
    umc_root=Path("/home/x-zjian1/umc")
    source_paths=[project_root/"py_codes/refresh_selected_results.py",project_root/"py_codes/benchmark_field_temporal.py",project_root/"py_codes/screen_field_geometric_order.py",project_root/"t1_start_c_code.py",umc_root/"include/compress.hpp",umc_root/"include/decompress.hpp",umc_root/"test/test_compress.cpp",umc_root/"test/test_decompress.cpp"]
    binary_paths=[project_root/"install/bin/s1_compress",project_root/"install/bin/s1_decompress",Path("/home/x-zjian1/umc/build/test/test_compress"),Path("/home/x-zjian1/umc/build/test/test_decompress")]
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=project_root,text=True,stdout=subprocess.PIPE,check=True).stdout.strip()
    diff=subprocess.run(["git","diff","--binary"],cwd=project_root,stdout=subprocess.PIPE,check=True).stdout
    umc_head=subprocess.run(["git","rev-parse","HEAD"],cwd=umc_root,text=True,stdout=subprocess.PIPE,check=True).stdout.strip()
    umc_diff=subprocess.run(["git","diff","--binary"],cwd=umc_root,stdout=subprocess.PIPE,check=True).stdout
    manifest={"field":field,"hostname":socket.gethostname(),"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"slurm_array_task_id":os.environ.get("SLURM_ARRAY_TASK_ID"),"allocated_cpus":os.environ.get("SLURM_CPUS_PER_TASK"),"OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS"),"OPENBLAS_NUM_THREADS":os.environ.get("OPENBLAS_NUM_THREADS"),"MKL_NUM_THREADS":os.environ.get("MKL_NUM_THREADS"),"git_head":head,"tracked_diff_sha256":hashlib.sha256(diff).hexdigest(),"umc_git_head":umc_head,"umc_tracked_diff_sha256":hashlib.sha256(umc_diff).hexdigest(),"sources":{str(path):sha256(path) for path in source_paths},"binaries":{str(path):sha256(path) for path in binary_paths},"inputs":{str(path):{"bytes":path.stat().st_size,"sha256":sha256(path)} for path in input_paths},"prediction_structure":None if not dag_path.exists() else {"path":str(dag_path),"bytes":dag_path.stat().st_size,"sha256":sha256(dag_path)},"rows":rows}
    manifest_path=result_root_path/"manifests"/f"{field['key']}.json"
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    with manifest_path.open("w",encoding="utf-8") as file:
        json.dump(manifest,file,ensure_ascii=False,indent=2)

def run_field(field,result_root_path=result_root,rel_values=rel_eb_vec):
    result_root_path.mkdir(parents=True,exist_ok=True)
    rows=[]
    total_bytes=sum((data_root/field["folder"]/f"{field['variable']}.dat.{time_step}").stat().st_size for time_step in range(field["steps"]))
    for index,rel_eb in enumerate(rel_values):
        method_vec=("unsz","umc") if index%2==0 else ("umc","unsz")
        if "umc" not in field:
            method_vec=("unsz",)
        for method in method_vec:
            print(f"开始：field={field['key']},compressor={method},REL={rel_eb:g}",flush=True)
            row=run_unsz(field,rel_eb,result_root_path) if method=="unsz" else run_umc(field,rel_eb)
            row.update({"field":field["key"],"dataset":field["dataset"],"variable":field["variable"],"time_step_num":field["steps"],"input_bytes":total_bytes,"compression_decompression_time":row["compression_time"]+row["decompression_time"],"preprocessing_MB_s":total_bytes/row["preprocessing_time"]/1e6,"compression_MB_s":total_bytes/row["compression_time"]/1e6,"decompression_MB_s":total_bytes/row["decompression_time"]/1e6})
            rows.append(row)
            write_rows(result_root_path/"fields"/f"{field['key']}.csv",rows)
    write_manifest(field,result_root_path,rows)
    print(f"完成：{field['key']}，结果{result_root_path/'fields'/f'{field['key']}.csv'}",flush=True)
