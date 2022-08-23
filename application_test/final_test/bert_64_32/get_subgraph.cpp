#include<iostream>
#include<algorithm>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<cstdio>
#include<stdlib.h>
#include<vector>
#include<string.h>
#include<cstring>
using namespace std;
bool is_number(const string &s)
{
    if(s.length()==0)return false;
    for(int i=0;i<s.length();i++)
    {
        if(!(s[i]>='0'&&s[i]<='9'))
        {
            return false;
        }
    }
    return true;
}
const int MAXN=10000;
int cnt=0;
int head[MAXN];
int node_num=-1;//表示原图当中的节点数目，需要求解更新
struct node
{
    int to;
    int next;
}edge[MAXN*2+5];
void add(int u,int v)
{
    edge[cnt].to=v;
    edge[cnt].next=head[u];
    head[u]=cnt++;
}
string name[MAXN];//node_id到node_name的映射
vector<string>input[MAXN];//以字符串的形式保存每个节点的输入点的编号,这个是为了处理net.dense2.weight和cst1这样的特殊情况的，同样可以用来作为反向边回溯
bool is_Apoint[MAXN];//判断一个点是否是A类点
bool is_Bpoint[MAXN];//判断一个点是否是B类点
bool vis[MAXN];
/*----------------------------------------test-----------------------*/
void dfs(int u)//单向边dfs暂时不需要visit数组
{
    if(vis[u])return;
    vis[u]=true;
    //cout<<u<<" "<<name[u]<<endl;
    if(is_Apoint[u])
    {
        //cout<<"this is a A_point"<<endl;
    }
    else if(is_Bpoint[u])
    {
        //cout<<"this is a B_point"<<endl;
    }
    for(int i=0;i<input[u].size();i++)
    {
        //cout<<input[u][i]<<" ";
    }
    //cout<<endl;
    for(int i=head[u];~i;i=edge[i].next)
    {
        int v=edge[i].to;
        dfs(v);
    }
}
/*--------------------------------------------test--------------------------*/
vector<int>A_point;//A类点集合，fullgraph的主体组成成分
vector<int>A_point_pre[MAXN];//fullgraph当每个A类点的前驱节点编号
void merge_dfs(int u,int root)//u一定是root的前向边
{
    if(vis[u])return;
    vis[u]=true;
    if(is_Apoint[u])
    {
        //在u和root之间建立新边，（其实如果可能的话，建立新边的时候动态删除相关旧边可以使得后续操作的复杂度大大降低，但是我们这里就不这么做了）
        //直接确定每个A类点的直接前驱
        A_point_pre[root].push_back(u);
        return;
    }
    else if(is_Bpoint[u])//对于B类点同样操作
    {
        A_point_pre[root].push_back(u);
        return;
    }
    //C类点就跳过，继续往前操作
    for(int i=0;i<input[u].size();i++)
    {
        if(is_number(input[u][i]))
        {
            stringstream ss;
            ss<<input[u][i];
            int v;
            ss>>v;
            ss.clear();
            merge_dfs(v,root);
        }
    }
}
int main()
{
    memset(head,-1,sizeof(head));
    memset(is_Apoint,false,sizeof(is_Apoint));
    memset(is_Bpoint,false,sizeof(is_Bpoint));
    ifstream in("fullgraph.txt");
    string line;
    if(in)
    {
        int pre=0;
        while(getline(in, line))
        { 
            stringstream ss(line);
            int node_id;
            ss>>node_id;//节点的id
            if(node_id!=pre+1)//说明是最后一行,一般有运行时间的节点数都大于等于2吧
            {
                //首先需要知道，上面读取进来的node_name需要重新转化为数字，表示有时间的节点的id，表示A类点
                ss.clear();
                stringstream sss(line);
                int x;
                while(sss>>x)
                {
                    is_Apoint[x]=true;
                    A_point.push_back(x);
                }
                sss.clear();
            }
            else
            {
                pre++;
                node_num=max(node_num,node_id);//求出原图节点数量
                string node_name;
                ss>>node_name;//节点的名称
                ss.clear();
                bool Bflag=true;//用一个点的前驱点判断该点是否是B类点

                name[node_id]=node_name;    

                if(node_name.find("NPUAllocFloatStatus")!=string::npos)continue;//bert专属
                if(node_name.find("Default/GetNext")!=string::npos)continue;

                //if(node_name.length()>=24 && node_name.substr(node_name.length()-24,19)=="NPUAllocFloatStatus")continue;//bert专属
                //if(node_name.length()>=15 && node_name.substr(0,15)=="Default/GetNext")continue;//vit的getnext不在第一个，去掉node_name匹配Default/GetNext***的



                getline(in,line);//继续读取下一行
                stringstream sss(line);
                string x;
                while(sss>>x)//将输入点全部读完
                {
                    input[node_id].push_back(x);

                    if(is_number(x))//数字点，需要建立从x到node_id的单向边
                    {
                        Bflag=false;
                        stringstream ssss;
                        int u;
                        ssss<<x;
                        ssss>>u;
                        add(u,node_id);
                        ssss.clear();
                    }
                    else//感觉这种点也没必要专门建边了。比如net.dense2.weight和cst1
                    {
                        continue;
                    }
                }
                sss.clear();
                if(Bflag)
                {
                    is_Bpoint[node_id]=true;
                }
            }
        }
    }

    /*--------------------------------------------test--------------------------*/
    /*--------------------------------------------test--------------------------*/


    //然后开始建立新的图的过程,上面的过程应该都把图建好了吧，首先从1号点开始dfs搜索一下，并且打印每个点的is_Apoint，input，以及name信息：

    /*--------------------------------------------test--------------------------*/
    
    /*
    memset(vis,false,sizeof(vis));
    for(int i=1;i<=node_num;i++)
    {
        if(!vis[i])
        {
            dfs(i);
        }
    }
    */
    
    /*--------------------------------------------test--------------------------*/
    //求解参考《三月工作记录10》，对于每个A类点向前回溯建立新图，看看是否可行

    
    for(int i=1;i<=node_num;i++)
    {
        memset(vis,false,sizeof(vis));
        if(is_Apoint[i])
        {
            vis[i]=true;
            for(int j=0;j<input[i].size();j++)
            {
                if(is_number(input[i][j]))
                {
                    stringstream ss;//这里默认A类点的前驱都是有编号的点；
                    ss<<input[i][j];
                    int u;
                    ss>>u;
                    if(vis[u])continue;
                    merge_dfs(u,i);
                    ss.clear();
                }
            }
        }
    }

    //cout<<"-----------this is fullgraph--------------"<<endl;

    vector<int>A_point_out[MAXN];//新加上的功能，用于统计每个点的后向连接点
    

    for(int i=0;i<A_point.size();i++)
    {
        //cout<<A_point[i]<<" "<<name[A_point[i]]<<endl;
        for(int j=0;j<A_point_pre[A_point[i]].size();j++)
        {
            //cout<<A_point_pre[A_point[i]][j]<<" ";

            /*------------------------------*/
            int x=A_point_pre[A_point[i]][j];//前向点
            if(is_Apoint[x])//那么就把点A_point[i]设置为x点的后继
            {
                A_point_out[x].push_back(A_point[i]);
            }

        }
        //cout<<endl;
    }
    //能否把它们的属性也加上呢？


    ofstream out("subgraph.txt",ios::trunc);
    for(int i=0;i<A_point.size();i++)
    {
        out<<A_point[i]<<" "<<name[A_point[i]]<<endl;
        vector<int>tmp;
        tmp.clear();//用于对前向点的清理
        for(int j=0;j<A_point_pre[A_point[i]].size();j++)
        {
            int x=A_point_pre[A_point[i]][j];
            if(is_Apoint[x])
            {
                tmp.push_back(x);
            }
        }
        if(tmp.size()==0)
        {
            out<<0<<endl;
        }
        else out<<tmp.size()<<" ";
        for(int j=0;j<tmp.size();j++)
        {
            if(j==tmp.size()-1)
            {
                out<<tmp[j]<<endl;
            }
            else
            {
                out<<tmp[j]<<" ";
            }
        }

        if(A_point_out[A_point[i]].size()==0)
        {
            out<<0<<endl;
        }
        else out<<A_point_out[A_point[i]].size()<<" ";


        for(int j=0;j<A_point_out[A_point[i]].size();j++)
        {
            if(j==A_point_out[A_point[i]].size()-1)
            {
                out<<A_point_out[A_point[i]][j]<<endl;
            }
            else
            {
                out<<A_point_out[A_point[i]][j]<<" ";
            }
        }
    }


    return 0;
}