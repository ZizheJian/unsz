#pragma once

#include <cassert>
#include <map>
#include <queue>
#include <stack>
#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace std;

template <class T>
class huffman_encoder_class
{
    private:
        class Node 
        {
            public:
                uint64_t c;
                Node *p[2];

                Node(uint64_t c_=0,Node *lp=nullptr,Node *rp=nullptr)
                {
                    c=c_;
                    p[0]=lp;
                    p[1]=rp;
                }

                inline unsigned char isLeaf()
                { 
                    return p[0]==nullptr;
                }
        };

        class HuffmanTree 
        {
            private:
                unsigned char _constructed=0;

                unsigned char len=0;
                int vec=0;

                class cmp
                {
                    public:
                        bool operator()(const pair<uint64_t,uint64_t> &u,const pair<uint64_t,uint64_t> &v) 
                        {
                            return u.second==v.second?u.first>v.first:u.second>v.second;
                        }
                };

            public:
                void dfs_mp(Node *u)
                {
                    if (u->isLeaf())
                    {
                        mplen[u->c]=len;
                        mpcode[u->c]=vec;
                        limit=std::max(limit,len);
                        return;
                    }
                    ++len;
                    dfs_mp(u->p[0]);
                    --len;
                    vec^=1<<len++;
                    dfs_mp(u->p[1]);
                    vec^=1<<--len;
                }

                void dfs_vec(Node *u)
                {
                    if (u->isLeaf())
                    {
                        veclen[u->c]=len;
                        veccode[u->c]=vec;
                        limit=std::max(limit,len);
                        return;
                    }
                    ++len;
                    dfs_vec(u->p[0]);
                    --len;
                    vec^=1<<len++;
                    dfs_vec(u->p[1]);
                    vec^=1<<--len;
                }

                unsigned char usemp;
                // 0 : vec
                // 1 : mp

                std::vector<unsigned char> veclen;
                std::vector<int> veccode;
                unordered_map<uint64_t,unsigned char> mplen;
                unordered_map<uint64_t,int> mpcode;

                T offset;
                // minimum bits for T
                unsigned char mbft;
                unsigned char limit;

                void init()
                {
                    _constructed=0;
                    ht.clear();
                    veclen.clear();
                    veccode.clear();
                    mplen.clear();
                    mpcode.clear();
                    vecfreq.clear();
                    mpfreq.clear();

                    offset=0;
                    mbft=0;
                    root=0;
                    n=0;
                    maxval=0;
                    limit=0;
                }

                HuffmanTree()
                {
                    init();
                }

                uint64_t root;
                uint64_t n;//node数量
                T maxval;
                std::vector<Node> ht;
                std::vector<uint64_t> vecfreq;//以vector形式存储的freq
                unordered_map<uint64_t,uint64_t> mpfreq;//以map形式存储的freq

                void addElementInMap(uint64_t c,uint64_t freqc)
                {
                    assert(!_constructed);
                    ht.push_back(Node(c));
                    mpfreq[c]=freqc;
                    ++n;
                }

                void addElementInVector(uint64_t c,uint64_t freqc)
                {
                    assert(!_constructed);
                    ht.push_back(Node(c));
                    vecfreq[c]=freqc;
                    ++n;
                }

                void constructHuffmanTree()
                {
                    assert(!_constructed);

                    if (n==1 || maxval==1)
                    {
                        mbft=1;
                        maxval=1;
                        offset+=ht[0].c;
                        ht[0].c=0;
                        ht.push_back(Node(0,&ht[0],nullptr));
                        if (usemp) 
                        {
                            mplen[0]=1;
                            mpcode[0]=0;
                        }
                        else
                        {
                            veclen[0]=1;
                            veccode[0]=0;
                        }
                        limit=1;
                        setConstructed();
                        return;
                    }

                    mbft=1;
                    while ((1llu<<mbft)<static_cast<uint64_t>(maxval))
                        ++mbft;

                    priority_queue<pair<uint64_t,uint64_t>,vector<pair<uint64_t,uint64_t>>,cmp> q;

                    if (usemp)
                        for (uint64_t i=0;i<ht.size();i++)
                            q.push({i,mpfreq[ht[i].c]});
                    else
                        for (uint64_t i=0;i<ht.size();i++)
                            q.push({i,vecfreq[ht[i].c]});

                    while (q.size()>1)
                    {
                        uint64_t u=q.top().first;
                        uint64_t freq_u=q.top().second;
                        q.pop();
                        uint64_t v=q.top().first;
                        uint64_t freq_v=q.top().second;
                        q.pop();
                        ht.push_back(Node(0,&ht[u],&ht[v]));
                        q.push({ht.size()-1,freq_u+freq_v});
                    }

                    root=ht.size()-1;

                    if (usemp)
                        dfs_mp(&ht[root]);
                    else
                        dfs_vec(&ht[root]);

                    setConstructed();

                }

                unsigned char isConstructed() 
                {
                    return _constructed;
                }

                void setConstructed()
                {
                    _constructed=1;
                }
        };

        HuffmanTree tree;

    public:
        void preprocess_encode(const T* const bins,uint64_t num_bin)
        {
            tree.init();
            T __minval,__maxval;
            __minval=*bins;
            __maxval=*bins;
            for (uint64_t i=1;i<num_bin;i++)
            {
                __minval=min(__minval,*(bins+i));
                __maxval=max(__maxval,*(bins+i));
            }
            tree.offset=__minval;
            tree.maxval=__maxval-__minval+1;
            // bin的值密集时使用vector存储频次，稀疏时使用map存储频次
            if ((tree.maxval>=(1<<12) and num_bin<2*static_cast<uint64_t>(__maxval)) || tree.maxval>=(1<<28))
                tree.usemp=1;
            else
                tree.usemp=0;
            if (tree.usemp)
            {
                map<uint64_t,uint64_t> freq;
                for (uint64_t i=0;i<num_bin;i++)
                    ++freq[static_cast<uint64_t>(bins[i]-tree.offset)];
                tree.ht.reserve(freq.size()<<1);
                for (auto it:freq)
                    tree.addElementInMap(it.first,it.second);
            }
            else
            {
                tree.vecfreq.resize(tree.maxval);
                tree.veclen.resize(tree.maxval);
                tree.veccode.resize(tree.maxval);
                std::vector<uint64_t> freq(tree.maxval);
                for (uint64_t i=0;i<num_bin;i++)
                    ++freq[static_cast<uint64_t>(bins[i]-tree.offset)];
                tree.ht.reserve(freq.size()<<1);
                for (uint64_t i=0;i<static_cast<uint64_t>(tree.maxval);i++)
                    if (freq[i]) 
                        tree.addElementInVector(i,freq[i]);
            }
            tree.constructHuffmanTree();
        }

        uint64_t encode(const T *bins,uint64_t num_bin,unsigned char*& bytes)
        {
            if (tree.maxval==1)
            {
                int64ToBytes_bigEndian(bytes,num_bin^0x1234abcd);
                bytes+=8;
                return 8;
            }

            assert(tree.isConstructed());
            unsigned char* head=bytes;
            bytes+=8;

            uint64_t len=0;

            unsigned char mask=0;
            unsigned char index=0;

            if (tree.usemp)
                for (uint64_t i=0;i<num_bin;i++)
                {
                    const T& it=bins[i];
                    const unsigned char& len_i=tree.mplen[it-tree.offset];
                    const int& code_i=tree.mpcode[it-tree.offset];
                    len+=len_i;
                    writeBytes(bytes,code_i,len_i,mask,index);
                }
            else
                for (uint64_t i=0;i<num_bin;i++) 
                {
                    const T& it=bins[i];
                    const unsigned char& len_i=tree.veclen[it-tree.offset];
                    const int& code_i=tree.veccode[it-tree.offset];
                    len+=len_i;
                    writeBytes(bytes,code_i,len_i,mask,index);
                }
            
            writeBytesClearMask(bytes,mask,index);

            int64ToBytes_bigEndian(head,len^0x1234abcd);

            return bytes-head;
        }

        void decode(const unsigned char*& bytes,T *out,uint64_t targetLength)
        {
            if (tree.maxval==1)
            {
                uint64_t len=bytesToInt64_bigEndian(bytes)^0x1234abcd;
                bytes+=8;
                for (uint64_t i=0;i<len;i++)
                    out[i]=tree.offset;
                return;
            }

            assert(tree.isConstructed());

            uint64_t len=bytesToInt64_bigEndian(bytes)^0x1234abcd;
            bytes+=8;
            uint64_t outLen=0;

            uint64_t byteIndex=0;
            uint64_t i=0;
            unsigned char b;
            Node *u=&tree.ht[tree.root];
            auto offset=tree.offset;

            for (;i+8<len;i+=8,byteIndex++)
            {
                b=bytes[byteIndex];

                u=u->p[b&1];
                if (u->isLeaf())
                {
                    out[outLen++]=u->c+offset;
                    u=&tree.ht[tree.root];
                }
                u=u->p[(b>>1)&1];
                if (u->isLeaf())
                {
                    out[outLen++]=u->c+offset;
                    u=&tree.ht[tree.root];
                }
                u=u->p[(b>>2)&1];
                if (u->isLeaf())
                {
                    out[outLen++]=u->c+offset;
                    u=&tree.ht[tree.root];
                }
                u=u->p[(b>>3)&1];
                if (u->isLeaf())
                {
                    out[outLen++]=u->c+offset;
                    u=&tree.ht[tree.root];
                }
                u=u->p[(b>>4)&1];
                if (u->isLeaf())
                {
                    out[outLen++]=u->c+offset;
                    u=&tree.ht[tree.root];
                }
                u=u->p[(b>>5)&1];
                if (u->isLeaf())
                {
                    out[outLen++]=u->c+offset;
                    u=&tree.ht[tree.root];
                }
                u=u->p[(b>>6)&1];
                if (u->isLeaf())
                {
                    out[outLen++]=u->c+offset;
                    u=&tree.ht[tree.root];
                }
                u=u->p[(b>>7)&1];
                if (u->isLeaf())
                {
                    out[outLen++]=u->c+offset;
                    u=&tree.ht[tree.root];
                }
            }

            b=bytes[byteIndex];

            for (uint64_t j=0;j<len-i;j++)
            {
                u=u->p[(b>>j)&1];
                if (u->isLeaf())
                {
                    out[outLen++]=u->c+tree.offset;
                    u=&tree.ht[tree.root];
                }
            }

            bytes+=(len+7)>>3;
        }

        void save(unsigned char*& c) 
        {
            saveAsDFSOrder(c);
        }

        void load(const unsigned char*& c)
        {
            loadAsDFSOrder(c);
        }

    private:
        inline void int64ToBytes_bigEndian(unsigned char *b,uint64_t num)
        {
            b[0]=static_cast<unsigned char>(num>>56);
            b[1]=static_cast<unsigned char>(num>>48);
            b[2]=static_cast<unsigned char>(num>>40);
            b[3]=static_cast<unsigned char>(num>>32);
            b[4]=static_cast<unsigned char>(num>>24);
            b[5]=static_cast<unsigned char>(num>>16);
            b[6]=static_cast<unsigned char>(num>>8);
            b[7]=static_cast<unsigned char>(num);
        }

        inline int64_t bytesToInt64_bigEndian(const unsigned char *b)
        {
            int64_t temp=0;
            int64_t res=0;

            res<<=8;
            temp=b[0]&0xff;
            res|=temp;

            res<<=8;
            temp=b[1]&0xff;
            res|=temp;

            res<<=8;
            temp=b[2]&0xff;
            res|=temp;

            res<<=8;
            temp=b[3]&0xff;
            res|=temp;

            res<<=8;
            temp=b[4]&0xff;
            res|=temp;

            res<<=8;
            temp=b[5]&0xff;
            res|=temp;

            res<<=8;
            temp=b[6]&0xff;
            res|=temp;

            res<<=8;
            temp=b[7]&0xff;
            res|=temp;

            return res;
        }

        static void writeBytesBit(unsigned char*& c,unsigned char val,unsigned char& mask,unsigned char& index) 
        {
            assert(val==0 || val==1);

            mask|=val<<index++;
            if (index==8) {
                *c++=mask;
                mask=index=0;
            }
        }

        static void writeBytes(unsigned char*& c,T val,unsigned char len,unsigned char& mask,unsigned char& index)
        {
            assert(len>=1 and len<=sizeof(T)*8);

            if (len+index>=8)
            {
                mask|=(val&((1<<(8-index))-1))<<index;
                val>>=8-index;
                len-=8-index;
                *c++=mask;
                mask=index=0;

                while (len >= 8)
                {
                    *c++=val&((1<<8)-1);
                    val>>=8;
                    len-=8;
                }
            }

            mask|=(val&((1<<len)-1))<<index;
            index+=len;
        }

        static void writeBytesByte(unsigned char*& c,unsigned char val)
        { 
            *c++=val;
        }

        static void writeBytesClearMask(unsigned char*& c,unsigned char& mask,unsigned char& index)
        {
            if (index>0)
                *c++=mask;
        }

        static unsigned char readBit(const unsigned char* const &c,int i)
        { 
            return ((*(c+(i>>3)))>>(i&7))&1;
        }

        void saveAsDFSOrder(unsigned char*& c)
        {
            unsigned char mask=0,index=0;

            writeBytesByte(c,(tree.usemp<<7)|((tree.n==1)<<6)|tree.mbft);
            writeBytes(c,tree.offset,sizeof(T)<<3,mask,index);
            int64ToBytes_bigEndian(c,tree.n);
            c+=sizeof(uint64_t);
            int64ToBytes_bigEndian(c,tree.maxval);
            c+=sizeof(uint64_t);

            if (tree.n==0 || tree.n==1)
            {
                writeBytesClearMask(c,mask,index);
                return;
            }

            std::stack<Node*> stk;

            stk.push(&tree.ht[tree.root]);

            while (!stk.empty())
            {
                Node* u=stk.top();
                stk.pop();
                if (u->isLeaf())
                {
                    writeBytesBit(c,0x01,mask,index);
                    writeBytes(c,u->c,tree.mbft,mask,index);
                }
                else
                {
                    writeBytesBit(c,0x00,mask,index);
                    stk.push(u->p[1]);
                    stk.push(u->p[0]);
                }
            }

            writeBytesClearMask(c,mask,index);
        }

        void loadAsDFSOrder(const unsigned char*& bytes)
        {
            tree.init();

            tree.usemp=(*bytes)>>7;
            tree.mbft=(*bytes)&0x3f;
            ++bytes;

            for (uint64_t i=0;i<sizeof(T);i++)
            {
                tree.offset|=static_cast<T>(*bytes)<<(i<<3);
                ++bytes;
            }

            tree.n=bytesToInt64_bigEndian(bytes);
            bytes+=sizeof(uint64_t);
            tree.ht.reserve(tree.n<<1);
            
            tree.maxval=bytesToInt64_bigEndian(bytes);
            bytes+=sizeof(uint64_t);
            if (tree.usemp==0x00)
                if (tree.n>0)
                {
                    tree.veccode.resize(tree.maxval);
                    tree.veclen.resize(tree.maxval);
                }

            if (tree.n==0)
            {
                tree.setConstructed();
                return;
            }

            if (tree.n==1)
            {
                tree.ht.resize(2);
                tree.root=0;
                tree.ht[0]=Node(0,&tree.ht[1]);
                tree.ht[1]=Node(0);
                if (tree.usemp)
                {
                    tree.mplen[0]=1;
                    tree.mpcode[0]=0;
                }
                else
                {
                    tree.veclen[0]=1;
                    tree.veccode[0]=0;
                }
                tree.setConstructed();
                return;
            }

            tree.root=0;
            tree.ht.push_back(Node());
            std::stack<Node*> stk;
            stk.push(&tree.ht[0]);
            uint64_t i=1;

            while (!stk.empty())
            {
                Node* u=stk.top();

                if (readBit(bytes,i++)==0x00)
                {
                    tree.ht.push_back(Node());
                    if (u->p[0]==nullptr)
                        u->p[0]=&tree.ht[tree.ht.size()-1];
                    else
                        u->p[1]=&tree.ht[tree.ht.size()-1];
                    stk.push(&tree.ht[tree.ht.size()-1]);
                }
                else
                {
                    T c=0;
                    for (int j=0;j<tree.mbft;j++)
                        c|=static_cast<T>(readBit(bytes,i++))<<j;
                    tree.ht.push_back(Node(c));
                    if (u->p[0]==nullptr)
                        u->p[0]=&tree.ht[tree.ht.size()-1];
                    else
                        u->p[1]=&tree.ht[tree.ht.size()-1];
                    while (!stk.empty() and stk.top()->p[1] != nullptr)
                        stk.pop();
                }
            }
            bytes+=(i+7)>>3;

            if (tree.usemp)
                tree.dfs_mp(&tree.ht[tree.root]);
            else
                tree.dfs_vec(&tree.ht[tree.root]);

            tree.setConstructed();
        }
};