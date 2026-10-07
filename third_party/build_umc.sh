#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
umc_root="$project_root/third_party/umc"
replacement_root="$project_root/third_party/umc_replacements"
upstream_url="https://github.com/xuanwu02/umc.git"
upstream_commit="52a86589de096cadfd8ab83719dc4c3e697e5886"
replacement_paths=(include/compress.hpp include/decompress.hpp parallel_src/mpas_parutils.hpp parallel_src/test_par_compress.cpp parallel_src/test_par_decompress.cpp)

if [[ ! -d "$umc_root/.git" && ! -f "$umc_root/.git" ]];then
    echo "没有找到UMC子模块，请先执行git submodule update --init third_party/umc" >&2
    exit 1
fi
if [[ "$(git -C "$umc_root" remote get-url origin)" != "$upstream_url" ]];then
    echo "UMC origin不是预期的官方仓库：$(git -C "$umc_root" remote get-url origin)" >&2
    exit 1
fi
git -C "$umc_root" fetch --quiet origin main
if ! git -C "$umc_root" cat-file -e "$upstream_commit^{commit}";then
    echo "官方UMC中没有找到固定commit：$upstream_commit" >&2
    exit 1
fi
if [[ "$(git -C "$umc_root" rev-parse HEAD)" != "$upstream_commit" ]];then
    echo "UMC没有处于固定commit：$upstream_commit" >&2
    exit 1
fi

mapfile -t changed_paths < <(git -C "$umc_root" diff --name-only "$upstream_commit" --)
for changed_path in "${changed_paths[@]}";do
    recognized=false
    for replacement_path in "${replacement_paths[@]}";do
        if [[ "$changed_path" == "$replacement_path" ]];then
            recognized=true
            if ! cmp -s "$replacement_root/$replacement_path" "$umc_root/$replacement_path";then
                echo "UMC包含未知源码修改，停止构建：$changed_path" >&2
                exit 1
            fi
            break
        fi
    done
    if [[ "$recognized" == false ]];then
        echo "UMC包含未知源码修改，停止构建：$changed_path" >&2
        exit 1
    fi
done
for path in "${replacement_paths[@]}";do
    cp "$replacement_root/$path" "$umc_root/$path"
done
if (( ${#changed_paths[@]}==${#replacement_paths[@]} ));then
    echo "UMC替换文件已经安装"
else
    echo "已安装UNSZ保存的UMC替换文件"
fi

if [[ ! -d "$umc_root/external/SZ3/install" ]];then
    echo "UMC依赖尚未安装，请先在third_party/umc中执行官方build_script.sh" >&2
    exit 1
fi

cmake -S "$umc_root" -B "$umc_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$umc_root/build" --target test_compress test_decompress test_par_compress test_par_decompress test_partition --parallel "${1:-8}"
