import os,re,tempfile
from third_party.run_command.run_command import run_command

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/LES"
field_list=["velocity_x","velocity_y","velocity_z"]
rel_eb=0.01

def main():
    with tempfile.TemporaryDirectory(prefix="unsz_les_",dir="/tmp") as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        os.symlink(os.path.join(data_dir,"conn.dat"),os.path.join(test_dir,"conn.dat"))
        for time_step,field in enumerate(field_list):
            os.symlink(os.path.join(data_dir,f"{field}.dat.0"),os.path.join(test_dir,f"les_velocity.dat.{time_step}"))
        time_step_expression=f"[0,{len(field_list)-1}]"
        unsz_arguments=f"-data-dir {test_dir} -variable les_velocity -time-step '{time_step_expression}' -M REL {rel_eb} -reverse-mean-distance-only-order-experimental"
        unsz_output=run_command(f"{project_root}/install/bin/s1_compress {unsz_arguments}")
        run_command(f"{project_root}/install/bin/s1_decompress {unsz_arguments}")
        unsz_result=[(float(br),float(psnr)) for br,psnr in re.findall(r"^cr=\s*[^,]+,\s*br=\s*([^,]+),\s*psnr=\s*([^\s]+)",unsz_output,re.MULTILINE)]
        umc_output=run_command(f"/home/x-zjian1/umc/build/test/test_compress {test_dir}/ les_velocity coords.dat conn.dat 0 3 2 1 {len(field_list)} 0 {rel_eb} 10")
        run_command(f"/home/x-zjian1/umc/build/test/test_decompress {test_dir}/ les_velocity coords.dat conn.dat 0 3 2 1 {len(field_list)}")
        umc_bytes=[int(value) for value in re.findall(r"compressed_size\s*=\s*(\d+)",umc_output)]
        umc_psnr=[float(value) for value in re.findall(r"^PSNR\s*=\s*([^\s]+)",umc_output,re.MULTILINE)]
        point_num=os.path.getsize(os.path.join(data_dir,"velocity_x.dat.0"))//4
        if len(unsz_result)!=len(field_list) or len(umc_bytes)!=len(field_list) or len(umc_psnr)!=len(field_list):
            raise RuntimeError(f"结果数量错误：UNSZ={len(unsz_result)}，UMC bytes={len(umc_bytes)}，UMC PSNR={len(umc_psnr)}")
        print("LES结果汇总：")
        for field,(unsz_br,unsz_psnr),compressed_bytes,psnr in zip(field_list,unsz_result,umc_bytes,umc_psnr):
            umc_br=compressed_bytes*8/point_num
            print(f"{field}: UNSZ BR={unsz_br:.6f},PSNR={unsz_psnr:.6f}; UMC BR={umc_br:.6f},PSNR={psnr:.6f}")

if __name__=="__main__":
    main()
