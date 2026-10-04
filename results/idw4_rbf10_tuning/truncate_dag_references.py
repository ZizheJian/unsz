import argparse,os,struct
from pathlib import Path

header_struct=struct.Struct("<QIIII")
pair_struct=struct.Struct("<II")
uint32_struct=struct.Struct("<I")
expected_magic=0x4452505A534E55

def read_exact(file,byte_num):
    data=file.read(byte_num)
    if len(data)!=byte_num:
        raise RuntimeError("DAG缓存提前结束")
    return data

def main():
    parser=argparse.ArgumentParser(description="保留已有DAG顺序并截取每个目标最近的前N个参考点")
    parser.add_argument("source",type=Path)
    parser.add_argument("destination",type=Path)
    parser.add_argument("max_reference_num",type=int)
    arguments=parser.parse_args()
    if arguments.max_reference_num<=0:
        raise ValueError("max_reference_num必须为正数")
    arguments.destination.parent.mkdir(parents=True,exist_ok=True)
    temporary_path=arguments.destination.with_suffix(arguments.destination.suffix+".tmp")
    with arguments.source.open("rb") as source,temporary_path.open("wb") as destination:
        header=read_exact(source,header_struct.size)
        magic,version,point_num,dim,round_num=header_struct.unpack(header)
        if magic!=expected_magic or version!=1:
            raise RuntimeError("不支持的DAG缓存格式")
        destination.write(header)
        round_counts=read_exact(source,round_num*uint32_struct.size)
        destination.write(round_counts)
        target_num=sum(struct.unpack(f"<{round_num}I",round_counts))
        if target_num!=point_num:
            raise RuntimeError("DAG缓存目标数量与点数不一致")
        for _ in range(target_num):
            target_id,reference_num=pair_struct.unpack(read_exact(source,pair_struct.size))
            references=read_exact(source,reference_num*uint32_struct.size)
            retained_num=min(reference_num,arguments.max_reference_num)
            destination.write(pair_struct.pack(target_id,retained_num))
            destination.write(references[:retained_num*uint32_struct.size])
        if source.read(1):
            raise RuntimeError("DAG缓存末尾存在额外数据")
    os.replace(temporary_path,arguments.destination)
    print(f"生成{arguments.destination}，点数={point_num}，维数={dim}，最大参考点数={arguments.max_reference_num}")

if __name__=="__main__":
    main()
