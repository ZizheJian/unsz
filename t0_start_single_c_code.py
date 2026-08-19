import os,sys
from third_party.run_command.run_command import run_command

silent_build=False
code_name_map={1:"s1_compress",2:"s2_read",3:"s3_sz3_style_compress"}

test_num=int(sys.argv[1])
code_name=code_name_map[test_num]
arguments=" ".join(sys.argv[2:])
os.makedirs("build",exist_ok=True)
run_command(f"cmake -S c_codes -B build -DCMAKE_INSTALL_PREFIX=install -DCODE_NAME={code_name}",silent=silent_build)
run_command("cmake --build build",silent=silent_build)
run_command("cmake --install build",silent=silent_build)
run_command(f"./install/bin/{code_name} {arguments}")
