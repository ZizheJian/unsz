#include <string>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;

int main()
{
    string data_path="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/attr0.dat.120";
    string conn_path="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/conn.dat";
    string coord_path="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/coords.dat";

    // string data_path="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean/layerThickness.dat.15";
    // string conn_path="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean/conn.dat";
    // string coord_path="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean/coords.dat";

    int dim=3;
    int node_num=0;
    vector<float> coord_vec;
    vector<vector<int>> adj_vec_vec(node_num);
    vector<unordered_set<int>> adj_vec_set(node_num);
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
    //////// 读conn_path ////////
    {
        ifstream conn_file(conn_path,ios::binary);
        vector<int> conn(node_num*(dim+1));
        conn_file.read((char*)conn.data(),node_num*(dim+1)*sizeof(int));
        conn_file.close();
        adj_vec_vec.resize(node_num);
        adj_vec_set.resize(node_num);
        for (int i=0;i<node_num*(dim+1);i+=dim+1)
        {
            adj_vec_set[conn[i]].insert(conn[i+1]);
            adj_vec_set[conn[i]].insert(conn[i+2]);
            adj_vec_set[conn[i]].insert(conn[i+3]);
            adj_vec_set[conn[i+1]].insert(conn[i]);
            adj_vec_set[conn[i+1]].insert(conn[i+2]);
            adj_vec_set[conn[i+1]].insert(conn[i+3]);
            adj_vec_set[conn[i+2]].insert(conn[i]);
            adj_vec_set[conn[i+2]].insert(conn[i+1]);
            adj_vec_set[conn[i+2]].insert(conn[i+3]);
            adj_vec_set[conn[i+3]].insert(conn[i]);
            adj_vec_set[conn[i+3]].insert(conn[i+1]);
            adj_vec_set[conn[i+3]].insert(conn[i+2]);
        }
        for (int i=0;i<node_num;i++)
            adj_vec_vec[i].assign(adj_vec_set[i].begin(),adj_vec_set[i].end());
    }
    //////// 读数据 ////////
    {
        ifstream data_file(data_path,ios::binary);
        data_vec.resize(node_num);
        data_file.read((char*)data_vec.data(),node_num*sizeof(float));
        data_file.close();
    }

    //////// 坐标范围 ////////
    vector<float> max_coord(dim,-numeric_limits<float>::infinity()),min_coord(dim,numeric_limits<float>::infinity());
    for (int i=0;i<node_num;i++)
        for (int d=0;d<dim;d++)
        {
            max_coord[d]=max(max_coord[d],coord_vec[i*dim+d]);
            min_coord[d]=min(min_coord[d],coord_vec[i*dim+d]);
        }
    
    //////// 统计相邻点之间的距离的分布 ////////
    float max_adj_dist=0,avg_adj_dist=0;
    int adj_num=0;
    for (int i=0;i<node_num;i++)
        for (int adj:adj_vec_vec[i])
        {
            float dist=0;
            for (int d=0;d<dim;d++)
                dist+=(coord_vec[i*dim+d]-coord_vec[adj*dim+d])*(coord_vec[i*dim+d]-coord_vec[adj*dim+d]);
            dist=sqrt(dist);
            max_adj_dist=max(max_adj_dist,dist);
            avg_adj_dist+=dist;
            adj_num++;
        }
    avg_adj_dist/=adj_num;
    printf("coord_range= [[%f,%f],[%f,%f],[%f,%f]]\n",min_coord[0],max_coord[0],min_coord[1],max_coord[1],min_coord[2],max_coord[2]);
    printf("max_adj_dist= %f\n",max_adj_dist);
    printf("avg_adj_dist= %f\n",avg_adj_dist);

    //////// 统计点的度的分布 ////////
    {
        vector<int> degree_count(20);
        for (int i=0;i<node_num;i++)
            if (adj_vec_vec[i].size()<degree_count.size()-1)
                degree_count[adj_vec_vec[i].size()]++;
            else
                degree_count.back()++;
        for (int i=0;i<degree_count.size();i++)
            if (degree_count[i]>0)
                if (i<degree_count.size()-1)
                    printf("degree=%d, count=%d\n",i,degree_count[i]);
                else
                    printf("degree>=%d, count=%d\n",degree_count.size()-1,degree_count.back());
    }
}