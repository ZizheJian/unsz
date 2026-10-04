import subprocess
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
kernels=("gaussian","exponential","matern32","matern52","inverse_multiquadric")
for kernel in kernels:
    command=[
        project_root/"install/bin/s1_compress",
        "-point-set-config",project_root/"point_set_configs/Katrina_attr0_rbf_rel_1e-3.json",
        "-predictor","rbf","-data-dir",data_dir,"-field","attr0","-time-step","[0,161]","-coordinate-dim",3,
        "-prediction-min-ref-num",10,"-prediction-max-ref-num",10,"-ordering-scope","global","-ordering-score-max-ref-num",1,
        "-ordering-score-min-ref-num",1,"-ordering-candidate-num",16,"-rbf-kernel",kernel,"-rbf-shape",0.03,"-rbf-nugget",1e-6,
        "-eb-schedule","inverse_rank_log2",0.85,"-eb-schedule-log2-shift",0.5,"-eb-schedule-max-log2-exponent",7,
        "-dag-cache",project_root/"results/rbf_kernel_tuning/dag/katrina/attr0_rbf10.dag","-coordinate-transform","auto",
        "-label-value-execution","fused","-value-flow","constant_extrapolation_spatial_residual","-label-flow","constant_extrapolation",
    ]
    output=subprocess.run([str(value) for value in command],check=True,text=True,stdout=subprocess.PIPE).stdout
    statistics=[line for line in output.splitlines() if line.startswith("Batched RBF weights:")]
    if len(statistics)!=1:
        raise RuntimeError(f"{kernel}应输出一行RBF权重统计，实际为{len(statistics)}")
    print(f"{kernel}: {statistics[0]}",flush=True)
