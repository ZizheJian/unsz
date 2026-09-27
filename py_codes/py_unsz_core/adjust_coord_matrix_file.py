from pathlib import Path
import numpy as np

matrix_file_magic="UNSZ_ADJUST_COORD_MATRIX"
matrix_file_version=2
training_type_vec=("spatial","spatiotemporal")

def write_adjust_coord_matrix(path,matrix,training_type,field):
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1]:
        raise ValueError("坐标变换矩阵必须是方阵")
    if training_type not in training_type_vec:
        raise ValueError(f"training_type必须是{'或'.join(training_type_vec)}")
    if not np.isfinite(matrix).all():
        raise ValueError("坐标变换矩阵必须全部是有限数")
    if np.any(matrix[np.triu_indices(matrix.shape[0],1)]!=0):
        raise ValueError("坐标变换矩阵必须是下三角矩阵")
    if any(character.isspace() for character in field):
        raise ValueError("field名称不能包含空白字符")
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    parameter=matrix[np.tril_indices(matrix.shape[0])]
    with path.open("w") as file:
        file.write(f"{matrix_file_magic} {matrix_file_version}\n")
        file.write(f"training_type {training_type}\n")
        file.write(f"field {field}\n")
        file.write(f"coordinate_dim {matrix.shape[0]}\n")
        file.write("lower_triangular "+" ".join(f"{value:.17g}" for value in parameter)+"\n")
    print(f"坐标变换矩阵已写入：{path}")

def read_adjust_coord_matrix(path):
    token=Path(path).read_text().split()
    if len(token)<2 or token[0]!=matrix_file_magic or int(token[1])!=matrix_file_version:
        raise ValueError(f"坐标变换矩阵文件格式不正确：{path}")
    item={}
    token_id=2
    while token_id<len(token):
        key=token[token_id]
        token_id+=1
        if key=="lower_triangular":
            if "coordinate_dim" not in item:
                raise ValueError("coordinate_dim必须位于lower_triangular之前")
            parameter_num=item["coordinate_dim"]*(item["coordinate_dim"]+1)//2
            item[key]=np.asarray(token[token_id:token_id+parameter_num],dtype=np.float64)
            token_id+=parameter_num
        else:
            if token_id>=len(token):
                raise ValueError(f"{key}后面缺少值")
            item[key]=int(token[token_id]) if key=="coordinate_dim" else token[token_id]
            token_id+=1
    for key in ("training_type","field","coordinate_dim","lower_triangular"):
        if key not in item:
            raise ValueError(f"坐标变换矩阵文件缺少{key}")
    if item["training_type"] not in training_type_vec:
        raise ValueError(f"未知的训练类型：{item['training_type']}")
    matrix=np.zeros((item["coordinate_dim"],item["coordinate_dim"]),dtype=np.float64)
    matrix[np.tril_indices(item["coordinate_dim"])]=item["lower_triangular"]
    item["matrix"]=matrix
    return item
