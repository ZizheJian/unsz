#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <queue>


using namespace std;

bool is_abnormal(float val)
{
    return val<-1000;
}

class block_tree_c
{
    public:
        int parent_id=-1;
        int left_child_id=-1,right_child_id=-1;
        int slice_dim=-1;
        float slice_coord=0;
        vector<vector<float>> coord;
        int index_begin=0,index_end=0;
};

void split_block(vector<block_tree_c> &block_vec,vector<int> &id_vec,vector<float> &coord_vec,int block_id,int dim)
{
    if (block_vec[block_id].index_end==block_vec[block_id].index_begin+1)
        return;
    block_vec[block_id].slice_dim=0;
    float max_len=block_vec[block_id].coord[0][1]-block_vec[block_id].coord[0][0];
    for (int d=1;d<block_vec[block_id].coord.size();d++)
    {
        float len=block_vec[block_id].coord[d][1]-block_vec[block_id].coord[d][0];
        if (len>max_len)
        {
            max_len=len;
            block_vec[block_id].slice_dim=d;
        }
    }
    block_vec[block_id].slice_coord=(block_vec[block_id].coord[block_vec[block_id].slice_dim][0]+block_vec[block_id].coord[block_vec[block_id].slice_dim][1])/2;
    int partition_pos=block_vec[block_id].index_begin;
    for (int i=block_vec[block_id].index_begin;i<block_vec[block_id].index_end;i++)
        if (coord_vec[id_vec[i]*dim+block_vec[block_id].slice_dim]<block_vec[block_id].slice_coord)
        {
            swap(id_vec[i],id_vec[partition_pos]);
            partition_pos++;
        }
    if (partition_pos!=block_vec[block_id].index_begin)
    {
        block_vec.push_back(block_tree_c());
        block_vec.back().parent_id=block_id;
        block_vec.back().coord=block_vec[block_id].coord;
        block_vec.back().coord[block_vec[block_id].slice_dim][1]=block_vec[block_id].slice_coord;
        block_vec.back().index_begin=block_vec[block_id].index_begin;
        block_vec.back().index_end=partition_pos;
        block_vec[block_id].left_child_id=block_vec.size()-1;
    }
    if (partition_pos!=block_vec[block_id].index_end)
    {
        block_vec.push_back(block_tree_c());
        block_vec.back().parent_id=block_id;
        block_vec.back().coord=block_vec[block_id].coord;
        block_vec.back().coord[block_vec[block_id].slice_dim][0]=block_vec[block_id].slice_coord;
        block_vec.back().index_begin=partition_pos;
        block_vec.back().index_end=block_vec[block_id].index_end;
        block_vec[block_id].right_child_id=block_vec.size()-1;
    }
}

float get_dist_to_node(float *coord1,float *coord2,int dim)
{
    float dist=0;
    for (int d=0;d<dim;d++)
        dist=dist+(coord1[d]-coord2[d])*(coord1[d]-coord2[d]);
    return dist;
}

float get_dist_to_block(float *coord,vector<vector<float>> &block_coord,int dim)
{
    float dist=0;
    for (int d=0;d<dim;d++)
    {
        if (coord[d]<block_coord[d][0])
            dist=dist+(coord[d]-block_coord[d][0])*(coord[d]-block_coord[d][0]);
        else if (coord[d]>block_coord[d][1])
            dist=dist+(coord[d]-block_coord[d][1])*(coord[d]-block_coord[d][1]);
    }
    return dist;
}

void find_nearest(priority_queue<pair<float,int>> &nearest_vec,int nearest_num,vector<block_tree_c> &block_vec,int block_id,vector<int> &id_vec,vector<float> &coord_vec,
    float *query_coord,int dim)
{
    if (block_vec[block_id].index_end==block_vec[block_id].index_begin+1)//block内有一个点，计算距离并和已有的nearest_vec比较
    {
        float dist=get_dist_to_node(query_coord,coord_vec.data()+id_vec[block_vec[block_id].index_begin]*dim,dim);
        if (nearest_vec.size()<nearest_num)
            nearest_vec.push({dist,block_vec[block_id].index_begin});
        else if (dist<nearest_vec.top().first)
        {
            nearest_vec.pop();
            nearest_vec.push({dist,block_vec[block_id].index_begin});
        }
    }
    else//block内有多个点，先访问包含query_coord的子块，再访问另一个子块并比较距离
    {
        int first_child_id=-1,second_child_id=-1;
        if (query_coord[block_vec[block_id].slice_dim]<block_vec[block_id].slice_coord)
        {
            first_child_id=block_vec[block_id].left_child_id;
            second_child_id=block_vec[block_id].right_child_id;
        }
        else
        {
            first_child_id=block_vec[block_id].right_child_id;
            second_child_id=block_vec[block_id].left_child_id;
        }
        if (first_child_id>=0)
            find_nearest(nearest_vec,nearest_num,block_vec,first_child_id,id_vec,coord_vec,query_coord,dim);
        if (second_child_id>=0)
        {
            float dist=get_dist_to_block(query_coord,block_vec[second_child_id].coord,dim);
            if (nearest_vec.size()<nearest_num||dist<nearest_vec.top().first)
                find_nearest(nearest_vec,nearest_num,block_vec,second_child_id,id_vec,coord_vec,query_coord,dim);
        }
    }
}

float lagrange_predict(float coord,vector<float> &ref_coord_vec,vector<float> &ref_value_vec)
{
    float pred=0;
    for (int j1=0;j1<ref_coord_vec.size();j1++)
    {
        float term=ref_value_vec[j1];
        for (int j2=0;j2<ref_coord_vec.size();j2++)
            if (j1!=j2)
                term*=(coord-ref_coord_vec[j2])/(ref_coord_vec[j1]-ref_coord_vec[j2]);
        pred=pred+term;
    }
    return pred;
}

void compress_point(vector<float> &compressed_data_vec,vector<float> &compressed_grid_data,vector<int> &qb_vec,int i,int grid_id_1d,int step1,int step2,vector<int> &grid_shape,
    int dim,int dim_id,vector<vector<pair<int,float>>> &grid_map,vector<float> &coord_vec,vector<float> &data_vec,float abs_eb,float grid_gap,vector<float> &min_coord)
{
    // 对于没有真实点的预测目标，直接对预测目标进行预测，并不做修正。
    // 对于有真实点的预测目标，对它的每个真实点单独做预测，并修正。预测目标本身则不需要预测。
    // 对于没有真实点的参考点，直接使用参考点的值。
    // 对于有真实点的参考点，使用距离预测目标最近的真实点的值进行预测。
    vector<float> temp_coord_vec,temp_value_vec;
    int grid_id_1d_left=grid_id_1d-step2;
    if (grid_map[grid_id_1d_left].size()>0)
    {
        int id=grid_map[grid_id_1d_left][0].first;
        temp_coord_vec.push_back(coord_vec[dim*id+dim_id]);
        temp_value_vec.push_back(compressed_data_vec[id]);
    }
    else
    {
        temp_coord_vec.push_back((i-step1)*grid_gap+min_coord[dim_id]);
        temp_value_vec.push_back(compressed_grid_data[grid_id_1d_left]);
    }
    if (i+step1<grid_shape[dim_id])//右侧的点没有越界
    {
        int grid_id_1d_right=grid_id_1d+step2;
        if (grid_map[grid_id_1d_right].size()>0)
        {
            int id=grid_map[grid_id_1d_right][0].first;
            temp_coord_vec.push_back(coord_vec[dim*id+dim_id]);
            temp_value_vec.push_back(compressed_data_vec[id]);
        }
        else
        {
            temp_coord_vec.push_back((i+step1)*grid_gap+min_coord[dim_id]);
            temp_value_vec.push_back(compressed_grid_data[grid_id_1d_right]);
        }
        
    }
    if (grid_map[grid_id_1d].size()>0)
        for (int i=0;i<grid_map[grid_id_1d].size();i++)
        {
            int id=grid_map[grid_id_1d][i].first;
            float coord=coord_vec[id*dim+dim_id];
            float value=data_vec[id];
            float pred=lagrange_predict(coord,temp_coord_vec,temp_value_vec);
            int qb=round((value-pred)/(2*abs_eb));
            qb_vec.push_back(qb);
            compressed_data_vec[id]=pred+qb*2*abs_eb;
        }
    else
    {
        float coord=i*grid_gap+min_coord[0];
        compressed_grid_data[grid_id_1d]=lagrange_predict(coord,temp_coord_vec,temp_value_vec);
    }
}

int main()
{
    string data_path="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/attr0.dat.80";
    string coord_path="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/coords.dat";
    float rel_eb=1e-3;

    int dim=3;
    int node_num=0;
    vector<float> coord_vec;
    vector<float> data_vec(node_num);
    //////// 读coord_path ////////
    {
        ifstream coord_file(coord_path,ios::binary);
        coord_file.seekg(0,ios::end);
        int num=coord_file.tellg()/sizeof(float);
        if (num%dim!=0)
            throw runtime_error("coord file 有"+to_string(num)+"个浮点数，无法被"+to_string(dim)+"整除");
        coord_file.seekg(0,ios::beg);
        node_num=num/dim;
        coord_vec.resize(node_num*dim);
        coord_file.read((char*)coord_vec.data(),node_num*dim*sizeof(float));
        coord_file.close();
    }
    //////// 读数据 ////////
    {
        ifstream data_file(data_path,ios::binary);
        data_vec.resize(node_num);
        data_file.read((char*)data_vec.data(),node_num*sizeof(float));
        data_file.close();
    }
    //////// 去除异常值 ////////
    {
        int pos1=0,pos2=0;
        for (pos2=0;pos2<node_num;pos2++)
            if (!is_abnormal(data_vec[pos2]))
            {
                data_vec[pos1]=data_vec[pos2];
                for (int d=0;d<dim;d++)
                    coord_vec[pos1*dim+d]=coord_vec[pos2*dim+d];
                pos1++;
            }
        data_vec.resize(pos1);
        coord_vec.resize(pos1*dim);
        node_num=pos1;
        printf("node_num=%d\n",node_num);
    }

    //////// 坐标范围 ////////
    vector<float> max_coord(dim,-numeric_limits<float>::infinity()),min_coord(dim,numeric_limits<float>::infinity());
    for (int i=0;i<node_num;i++)
        for (int d=0;d<dim;d++)
        {
            max_coord[d]=max(max_coord[d],coord_vec[i*dim+d]);
            min_coord[d]=min(min_coord[d],coord_vec[i*dim+d]);
        }
    printf("Coord range: [[%f, %f], [%f, %f], [%f, %f]]\n",min_coord[0],max_coord[0],min_coord[1],max_coord[1],min_coord[2],max_coord[2]);
    //////// 数值范围 ////////
    float max_data=-numeric_limits<float>::infinity(),min_data=numeric_limits<float>::infinity();
    for (int i=0;i<node_num;i++)
    {
        max_data=max(max_data,data_vec[i]);
        min_data=min(min_data,data_vec[i]);
    }
    printf("Data range: [%f, %f]\n",min_data,max_data);
    float abs_eb=rel_eb*(max_data-min_data);

    //////// 构建kd-tree并计算最近邻距离 ////////
    // vector<int> id_vec(node_num);
    // for (int i=0;i<node_num;i++)
    //     id_vec[i]=i;
    // vector<block_tree_c> block_vec;
    // block_vec.push_back(block_tree_c());
    // block_vec[0].parent_id=-1;
    // block_vec[0].left_child_id=-1;
    // block_vec[0].right_child_id=-1;
    // block_vec[0].slice_dim=-1;
    // block_vec[0].slice_coord=0;
    // block_vec[0].coord.resize(dim);
    // for (int d=0;d<dim;d++)
    // {
    //     block_vec[0].coord[d].resize(2);
    //     block_vec[0].coord[d][0]=min_coord[d];
    //     block_vec[0].coord[d][1]=max_coord[d];
    // }
    // block_vec[0].index_begin=0;
    // block_vec[0].index_end=node_num;
    // for (int i=0;i<block_vec.size();i++)
    //     split_block(block_vec,id_vec,coord_vec,i,dim);
    // float min_dist=0;
    // for (int i=0;i<dim;i++)
    //     min_dist=min_dist+(max_coord[i]-min_coord[i])*(max_coord[i]-min_coord[i]);
    // for (int i=0;i<node_num;i++)
    // {
    //     priority_queue<pair<float,int>> nearest_vec;
    //     find_nearest(nearest_vec,2,block_vec,0,id_vec,coord_vec,coord_vec.data()+i*dim,dim);
    //     min_dist=min(min_dist,nearest_vec.top().first);
    //     nearest_vec.pop();
    //     if (nearest_vec.top().first!=0)
    //         throw runtime_error("nearest_vec[1].first="+to_string(nearest_vec.top().first)+"不为0");
    // }
    // printf("Min dist: %f\n",sqrt(min_dist));

    //////// 记录每个网格点对应的真实点集合 ////////
    float grid_gap=1;
    for (int d=0;d<dim;d++)
        grid_gap*=(max_coord[d]-min_coord[d]);
    grid_gap/=node_num;
    grid_gap=pow(grid_gap,1.0/dim);
    grid_gap/=2;
    printf("Grid gap: %f\n",grid_gap);
    vector<int> grid_shape(dim);
    for (int d=0;d<dim;d++)
        grid_shape[d]=round((max_coord[d]-min_coord[d])/grid_gap)+1;
    printf("Grid shape: [%d, %d, %d]\n",grid_shape[0],grid_shape[1],grid_shape[2]);

    //////// 记录每个网格点对应的真实点集合 ////////
    vector<vector<pair<int,float>>> grid_map(grid_shape[0]*grid_shape[1]*grid_shape[2]);
    for (int i=0;i<node_num;i++)
    {
        vector<int> grid_id(dim);
        for (int d=0;d<dim;d++)
            grid_id[d]=round((coord_vec[i*dim+d]-min_coord[d])/grid_gap);
        int grid_id_1d=grid_id[0]*grid_shape[1]*grid_shape[2]+grid_id[1]*grid_shape[2]+grid_id[2];
        float grid_coord[3];
        for (int d=0;d<dim;d++)
            grid_coord[d]=grid_id[d]*grid_gap+min_coord[d];
        grid_map[grid_id_1d].push_back({i,get_dist_to_node(grid_coord,coord_vec.data()+i*dim,dim)});
    }
    for (auto &grid_id:grid_map)
        sort(grid_id.begin(),grid_id.end(),[](const pair<int,float> &a,const pair<int,float> &b){return a.second<b.second;});
    float overlap_rate=0;
    for (auto &grid_id:grid_map)
        if (grid_id.size()>1)
            overlap_rate+=grid_id.size()-1;
    overlap_rate/=node_num;
    printf("Overlap rate: %f\n",overlap_rate);

    vector<float> compressed_data_vec(node_num,0);
    int grid_step_power=0;
    for (int d=0;d<dim;d++)
        grid_step_power=max(grid_step_power,(int)log2(grid_shape[d]-1));
    vector<float> compressed_grid_data(grid_shape[0]*grid_shape[1]*grid_shape[2],numeric_limits<float>::lowest());
    vector<int> qb_vec;
    qb_vec.reserve(node_num);
    {// 压缩[0,0,0]点
        if (grid_map[0].size()>0)
            for (int i=0;i<grid_map[0].size();i++)
            {
                int id=grid_map[0][i].first;
                int qb=round(data_vec[id]/(2*abs_eb));
                qb_vec.push_back(qb);
                compressed_data_vec[id]=qb*2*abs_eb;
            }
        else
            compressed_grid_data[0]=0;//这里假设我们没有建立KDtree，所以没法找到距离[0,0,0]点最近的真实点来进行预测，所以只能直接把[0,0,0]点的压缩值设为0
    }
    for (;grid_step_power>=0;grid_step_power--)
    {
        int step=1<<grid_step_power;
        for (int i0=step;i0<grid_shape[0];i0+=2*step)//沿着0号维度预测
            for (int i1=0;i1<grid_shape[1];i1+=2*step)
                for (int i2=0;i2<grid_shape[2];i2+=2*step)
                    compress_point(compressed_data_vec,compressed_grid_data,qb_vec,i0,i0*grid_shape[1]*grid_shape[2]+i1*grid_shape[2]+i2,step,step*grid_shape[1]*grid_shape[2],
                        grid_shape,dim,0,grid_map,coord_vec,data_vec,abs_eb,grid_gap,min_coord);
        for (int i0=0;i0<grid_shape[0];i0+=step)//沿着1号维度预测
            for (int i1=step;i1<grid_shape[1];i1+=2*step)
                for (int i2=0;i2<grid_shape[2];i2+=2*step)
                    compress_point(compressed_data_vec,compressed_grid_data,qb_vec,i1,i0*grid_shape[1]*grid_shape[2]+i1*grid_shape[2]+i2,step,step*grid_shape[2],
                        grid_shape,dim,1,grid_map,coord_vec,data_vec,abs_eb,grid_gap,min_coord);
        for (int i0=0;i0<grid_shape[0];i0+=step)//沿着2号维度预测
            for (int i1=0;i1<grid_shape[1];i1+=step)
                for (int i2=step;i2<grid_shape[2];i2+=2*step)
                    compress_point(compressed_data_vec,compressed_grid_data,qb_vec,i2,i0*grid_shape[1]*grid_shape[2]+i1*grid_shape[2]+i2,step,step,
                        grid_shape,dim,2,grid_map,coord_vec,data_vec,abs_eb,grid_gap,min_coord);
    }
    printf("qb_vec.size()=%d\n",qb_vec.size());
    for (int i=0;i<node_num;i++)
    {
        float err=abs(compressed_data_vec[i]-data_vec[i]);
        if (err>1.001*abs_eb)
        {
            printf("i=%d, err=%f, abs_eb=%f\n",i,err,abs_eb);
            throw runtime_error("压缩后的值与原值的误差超过了绝对误差界限");
        }
    }
    
}