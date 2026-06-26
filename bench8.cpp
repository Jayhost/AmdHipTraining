// bench_gpus_v4.cpp
// Compile: g++ -O3 bench_gpus_v4.cpp -o bench_gpus_v4 -lvulkan -lshaderc_shared

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <random>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <array>
#include <chrono>

#define VK_CHECK(call) do { VkResult err = call; if(err!=VK_SUCCESS){fprintf(stderr,"VK err %s:%d: %d\n",__FILE__,__LINE__,err);exit(1);} }while(0)

const int T=64, C=128, NH=8, NKV=4, NGRP=NH/NKV, DK=C/NH, DFF=512, L=6;
const float RMS_EPS=1e-6f; const int BLOCK_SIZE=256;
const int MAX_GPU=3;

static const char* glsl_h = R"glsl(
float loadF(int idx){return uintBitsToFloat(fdata[idx]);}
void storeF(int idx,float val){fdata[idx]=floatBitsToUint(val);}
)glsl";

static const char* sh_enc = R"glsl(#version 450
layout(local_size_x=256)in;layout(set=0,binding=0)buffer FloatBuf{uint fdata[];};layout(set=0,binding=1)buffer IntBuf{int idata[];};
layout(std430,push_constant)uniform PC{int o,io,w,B,T,C,p1,p2,p3,p4,p5,p6,p7,p8,p9;}p;
void main(){int i=int(gl_GlobalInvocationID.x);int N=p.B*p.T*p.C;if(i>=N)return;int c=i%p.C,bt=i/p.C;int ix=idata[p.io+bt];storeF(p.o+i,loadF(p.w+ix*p.C+c));}
)glsl";

static const char* sh_rms = R"glsl(#version 450
layout(local_size_x=256)in;layout(set=0,binding=0)buffer FloatBuf{uint fdata[];};
layout(std430,push_constant)uniform PC{int o,rs,inp,w,B,T,C,p1,p2,p3,p4,p5,p6,p7,p8,p9;}p;
void main(){int i=int(gl_GlobalInvocationID.x);int N=p.B*p.T*p.C;if(i>=N)return;int c=i%p.C,bt=i/p.C;float ss=0.0;for(int j=0;j<p.C;j++){float x=loadF(p.inp+bt*p.C+j);ss+=x*x;}ss=ss/p.C+1e-6;float s=1.0/sqrt(ss);storeF(p.o+i,s*loadF(p.inp+bt*p.C+c)*loadF(p.w+c));}
)glsl";

static const char* sh_mm = R"glsl(#version 450
layout(local_size_x=256)in;layout(set=0,binding=0)buffer FloatBuf{uint fdata[];};
layout(std430,push_constant)uniform PC{int o,inp,w,b,tb,C,OC,hb,p1,p2,p3,p4,p5,p6,p7;}p;
void main(){int i=int(gl_GlobalInvocationID.x);int N=p.tb*p.OC;if(i>=N)return;int oo=i%p.OC,bt=i/p.OC;float v=(p.hb!=0)?loadF(p.b+oo):0.0;for(int j=0;j<p.C;j++)v+=loadF(p.inp+bt*p.C+j)*loadF(p.w+oo*p.C+j);storeF(p.o+i,v);}
)glsl";

static const char* sh_qkv = R"glsl(#version 450
layout(local_size_x=256)in;layout(set=0,binding=0)buffer FloatBuf{uint fdata[];};
layout(std430,push_constant)uniform PC{int oq,ok,ov,inp,wq,wk,wv,B,T,C,NH,NKV,DK,p1,p2,p3,p4;}p;
void main(){
    int i=int(gl_GlobalInvocationID.x);
    int BT=p.B*p.T;
    int total_q=BT*p.NH*p.DK;
    int total_qk=total_q+BT*p.NKV*p.DK;
    int total_qkv=total_qk+BT*p.NKV*p.DK;
    if(i>=total_qkv)return;
    int bt,out_d;float val=0.0;
    if(i<total_q){
        bt=i/(p.NH*p.DK);out_d=i%(p.NH*p.DK);
        for(int j=0;j<p.C;j++)val+=loadF(p.inp+bt*p.C+j)*loadF(p.wq+out_d*p.C+j);
        storeF(p.oq+i,val);
    }else if(i<total_qk){
        int ii=i-total_q;bt=ii/(p.NKV*p.DK);out_d=ii%(p.NKV*p.DK);
        for(int j=0;j<p.C;j++)val+=loadF(p.inp+bt*p.C+j)*loadF(p.wk+out_d*p.C+j);
        storeF(p.ok+ii,val);
    }else{
        int ii=i-total_qk;bt=ii/(p.NKV*p.DK);out_d=ii%(p.NKV*p.DK);
        for(int j=0;j<p.C;j++)val+=loadF(p.inp+bt*p.C+j)*loadF(p.wv+out_d*p.C+j);
        storeF(p.ov+ii,val);
    }
}
)glsl";

static const char* sh_q_norm_rope = R"glsl(#version 450
layout(local_size_x=256)in;layout(set=0,binding=0)buffer FloatBuf{uint fdata[];};
layout(std430,push_constant)uniform PC{int o,inp,wn,co,si,B,T,nh,dk,p1,p2,p3,p4,p5,p6,p7,p8;}p;
void main(){
    int i=int(gl_GlobalInvocationID.x);int N=p.B*p.T*p.nh*p.dk;if(i>=N)return;
    int d=i%p.dk;int h=(i/p.dk)%p.nh;int t=(i/(p.dk*p.nh))%p.T;int b=i/(p.dk*p.nh*p.T);
    if(d%2!=0)return;
    int base=b*p.T*p.nh*p.dk+t*p.nh*p.dk+h*p.dk;
    float ss=0.0;for(int j=0;j<p.dk;j++){float x=loadF(p.inp+base+j);ss+=x*x;}
    float s=1.0/sqrt(ss/p.dk+1e-6);
    float nd0=s*loadF(p.inp+base+d)*loadF(p.wn+h*p.dk+d);
    float nd1=s*loadF(p.inp+base+d+1)*loadF(p.wn+h*p.dk+d+1);
    int hi=d/2;int hd=p.dk/2;
    float cv=loadF(p.co+t*hd+hi);float sv=loadF(p.si+t*hd+hi);
    storeF(p.o+base+d,nd0*cv-nd1*sv);storeF(p.o+base+d+1,nd0*sv+nd1*cv);
}
)glsl";

static const char* sh_gqa_v2 = R"glsl(#version 450
layout(local_size_x=256) in;
layout(set=0, binding=0) buffer FloatBuf { uint fdata[]; };

layout(std430, push_constant) uniform PC {
    int o; int att; int qo; int ko; int vo;
    int B; int T; int NH; int NKV; int DK; int NGRP;
    int p1; int p2; int p3; int p4; int p5;
} p;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int total = p.B * p.T * p.NH * p.DK;
    if (i >= total) return;
    int d = i % p.DK;
    int h = (i / p.DK) % p.NH;
    int t = (i / (p.DK * p.NH)) % p.T;
    int b = i / (p.DK * p.NH * p.T);
    int kv = h / p.NGRP;
    float sc = 1.0 / sqrt(float(p.DK));

    int q_base = p.qo + b * p.T * p.NH * p.DK + t * p.NH * p.DK + h * p.DK;
    int k_b = p.ko + b * p.T * p.NKV * p.DK;
    int v_b = p.vo + b * p.T * p.NKV * p.DK;
    int att_bt = p.att + b * p.NH * p.T * p.T + h * p.T * p.T + t * p.T;

    float mx = -1e10;
    for (int t2 = 0; t2 <= t; t2++) {
        int k_t2 = k_b + t2 * p.NKV * p.DK + kv * p.DK;
        float val = 0.0;
        for (int j = 0; j < p.DK; j++) val += loadF(q_base + j) * loadF(k_t2 + j);
        val *= sc;
        if (val > mx) mx = val;
    }
    float es = 0.0;
    for (int t2 = 0; t2 <= t; t2++) {
        int k_t2 = k_b + t2 * p.NKV * p.DK + kv * p.DK;
        float val = 0.0;
        for (int j = 0; j < p.DK; j++) val += loadF(q_base + j) * loadF(k_t2 + j);
        val *= sc;
        float ev = exp(val - mx);
        es += ev;
        storeF(att_bt + t2, ev);
    }
    for (int t2 = t + 1; t2 < p.T; t2++) storeF(att_bt + t2, 0.0);
    float inv = (es == 0.0) ? 0.0 : 1.0 / es;
    float result = 0.0;
    for (int t2 = 0; t2 <= t; t2++) {
        int v_t2 = v_b + t2 * p.NKV * p.DK + kv * p.DK;
        float a = loadF(att_bt + t2) * inv;
        result += a * loadF(v_t2 + d);
    }
    int o_bt = p.o + b * p.T * p.NH * p.DK + t * p.NH * p.DK + h * p.DK;
    storeF(o_bt + d, result);
}
)glsl";


static const char* sh_res_rms = R"glsl(#version 450
layout(local_size_x=256)in;layout(set=0,binding=0)buffer FloatBuf{uint fdata[];};
layout(std430,push_constant)uniform PC{int o,i1,i2,w,B,T,C,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13;}p;
void main(){
    int i=int(gl_GlobalInvocationID.x);int N=p.B*p.T*p.C;if(i>=N)return;
    int c=i%p.C;int bt=i/p.C;
    float ss=0.0;for(int j=0;j<p.C;j++){float x=loadF(p.i1+bt*p.C+j)+loadF(p.i2+bt*p.C+j);ss+=x*x;}
    float s=1.0/sqrt(ss/p.C+1e-6);
    float val=loadF(p.i1+i)+loadF(p.i2+i);
    storeF(p.o+i,s*val*loadF(p.w+c));
}
)glsl";

static const char* sh_gate_up_swiglu = R"glsl(#version 450
layout(local_size_x=256)in;layout(set=0,binding=0)buffer FloatBuf{uint fdata[];};
layout(std430,push_constant)uniform PC{int o,inp,wg,wu,B,T,C,DFF,p1,p2,p3,p4,p5,p6,p7,p8,p9;}p;
void main(){
    int i=int(gl_GlobalInvocationID.x);int N=p.B*p.T*p.DFF;if(i>=N)return;
    int d=i%p.DFF;int bt=i/p.DFF;
    float g=0.0;float u=0.0;
    for(int j=0;j<p.C;j++){float v=loadF(p.inp+bt*p.C+j);g+=v*loadF(p.wg+d*p.C+j);u+=v*loadF(p.wu+d*p.C+j);}
    float sig=1.0/(1.0+exp(-g));
    storeF(p.o+i,g*sig*u);
}
)glsl";

static const char* sh_down_res = R"glsl(#version 450
layout(local_size_x=256)in;layout(set=0,binding=0)buffer FloatBuf{uint fdata[];};
layout(std430,push_constant)uniform PC{int o,sg,wd,res_prev,attproj,B,T,C,DFF,p1,p2,p3,p4,p5,p6,p7,p8;}p;
void main(){
    int i=int(gl_GlobalInvocationID.x);int N=p.B*p.T*p.C;if(i>=N)return;
    int c=i%p.C;int bt=i/p.C;
    float val=0.0;
    for(int j=0;j<p.DFF;j++)val+=loadF(p.sg+bt*p.DFF+j)*loadF(p.wd+c*p.DFF+j);
    val+=loadF(p.res_prev+i)+loadF(p.attproj+i);
    storeF(p.o+i,val);
}
)glsl";

// ============================================================
// Vulkan boilerplate
// ============================================================
struct Pipe {
    VkPipeline p;
    VkPipelineLayout l;
};
struct VCtx{VkInstance inst;VkPhysicalDevice pd;VkDevice dev;VkQueue q;VkCommandPool cp;VkCommandBuffer cb;VkDescriptorSetLayout dsl;VkDescriptorPool dp;VkDescriptorSet ds;VkFence fence;uint32_t qf;char name[256];};
struct Buf{VkBuffer b;VkDeviceMemory m;void*mp;VkDeviceSize sz;};

std::vector<uint32_t> comp(const char*src,const char*nm){std::string s=src;size_t p=s.find("void main()");if(p!=std::string::npos)s.insert(p,"\n"+std::string(glsl_h)+"\n");shaderc::Compiler c;shaderc::CompileOptions o;o.SetTargetEnvironment(shaderc_target_env_vulkan,shaderc_env_version_vulkan_1_2);auto r=c.CompileGlslToSpv(s.c_str(),s.length(),shaderc_compute_shader,nm,o);if(r.GetCompilationStatus()!=shaderc_compilation_status_success){fprintf(stderr,"Err %s: %s\n",nm,r.GetErrorMessage().c_str());exit(1);}return{r.cbegin(),r.cend()};}
VkShaderModule mkSM(const VCtx&c,const std::vector<uint32_t>&code){VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};ci.codeSize=code.size()*4;ci.pCode=code.data();VkShaderModule sm;VK_CHECK(vkCreateShaderModule(c.dev,&ci,0,&sm));return sm;}
uint32_t findMT(const VCtx&c,uint32_t tf,VkMemoryPropertyFlags p){VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(c.pd,&mp);for(uint32_t i=0;i<mp.memoryTypeCount;i++)if((tf&(1<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p)return i;fprintf(stderr,"Mem fail\n");exit(1);}
void mkBuf(const VCtx&c,VkDeviceSize sz,VkBufferUsageFlags u,VkMemoryPropertyFlags p,Buf&b){VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=sz;bi.usage=u;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK_CHECK(vkCreateBuffer(c.dev,&bi,0,&b.b));VkMemoryRequirements mr;vkGetBufferMemoryRequirements(c.dev,b.b,&mr);VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=mr.size;ai.memoryTypeIndex=findMT(c,mr.memoryTypeBits,p);VK_CHECK(vkAllocateMemory(c.dev,&ai,0,&b.m));VK_CHECK(vkBindBufferMemory(c.dev,b.b,b.m,0));if(p&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)VK_CHECK(vkMapMemory(c.dev,b.m,0,sz,0,&b.mp));else b.mp=nullptr;b.sz=sz;}
void destroyBuf(const VCtx&c,Buf&b){vkDestroyBuffer(c.dev,b.b,0);vkFreeMemory(c.dev,b.m,0);}
Pipe mkPipe(const VCtx&c,VkShaderModule sm){Pipe cp;VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};si.stage=VK_SHADER_STAGE_COMPUTE_BIT;si.module=sm;si.pName="main";VkPushConstantRange pcr{};pcr.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;pcr.size=16*sizeof(int);VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&c.dsl;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcr;VK_CHECK(vkCreatePipelineLayout(c.dev,&pli,0,&cp.l));VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pi.stage=si;pi.layout=cp.l;VK_CHECK(vkCreateComputePipelines(c.dev,VK_NULL_HANDLE,1,&pi,0,&cp.p));vkDestroyShaderModule(c.dev,sm,0);return cp;}
void initCtx(VCtx&c,VkInstance inst,VkPhysicalDevice pd){c.inst=inst;c.pd=pd;VkPhysicalDeviceProperties pr;vkGetPhysicalDeviceProperties(pd,&pr);strncpy(c.name,pr.deviceName,255);c.name[255]=0;uint32_t qfc=0;vkGetPhysicalDeviceQueueFamilyProperties(pd,&qfc,0);std::vector<VkQueueFamilyProperties> qf(qfc);vkGetPhysicalDeviceQueueFamilyProperties(pd,&qfc,qf.data());c.qf=0;for(uint32_t i=0;i<qfc;i++)if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){c.qf=i;break;}float qp=1.f;VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qci.queueFamilyIndex=c.qf;qci.queueCount=1;qci.pQueuePriorities=&qp;VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&qci;VK_CHECK(vkCreateDevice(pd,&dci,0,&c.dev));vkGetDeviceQueue(c.dev,c.qf,0,&c.q);VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;cpi.queueFamilyIndex=c.qf;VK_CHECK(vkCreateCommandPool(c.dev,&cpi,0,&c.cp));VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cba.commandPool=c.cp;cba.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cba.commandBufferCount=1;VK_CHECK(vkAllocateCommandBuffers(c.dev,&cba,&c.cb));VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VK_CHECK(vkCreateFence(c.dev,&fi,0,&c.fence));VkDescriptorSetLayoutBinding fbb{};fbb.binding=0;fbb.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;fbb.descriptorCount=1;fbb.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;VkDescriptorSetLayoutBinding ibb{};ibb.binding=1;ibb.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;ibb.descriptorCount=1;ibb.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;std::array<VkDescriptorSetLayoutBinding,2>bs={fbb,ibb};VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dli.bindingCount=2;dli.pBindings=bs.data();VK_CHECK(vkCreateDescriptorSetLayout(c.dev,&dli,0,&c.dsl));VkDescriptorPoolSize dps[]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}};VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=dps;VK_CHECK(vkCreateDescriptorPool(c.dev,&dpi,0,&c.dp));VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dsai.descriptorPool=c.dp;dsai.descriptorSetCount=1;dsai.pSetLayouts=&c.dsl;VK_CHECK(vkAllocateDescriptorSets(c.dev,&dsai,&c.ds));}
void bindBufs(const VCtx&c,const Buf&fb,const Buf&ib){VkDescriptorBufferInfo fbi{};fbi.buffer=fb.b;fbi.offset=0;fbi.range=fb.sz;VkDescriptorBufferInfo ibi{};ibi.buffer=ib.b;ibi.offset=0;ibi.range=ib.sz;VkWriteDescriptorSet fw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};fw.dstSet=c.ds;fw.dstBinding=0;fw.descriptorCount=1;fw.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;fw.pBufferInfo=&fbi;VkWriteDescriptorSet iw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};iw.dstSet=c.ds;iw.dstBinding=1;iw.descriptorCount=1;iw.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;iw.pBufferInfo=&ibi;std::array<VkWriteDescriptorSet,2>ws={fw,iw};vkUpdateDescriptorSets(c.dev,2,ws.data(),0,0);}
void beginC(const VCtx&c){VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;VK_CHECK(vkResetCommandBuffer(c.cb,0));VK_CHECK(vkBeginCommandBuffer(c.cb,&bi));}
void recC(const VCtx&c,Pipe p,int N,const std::vector<int>&pc){vkCmdBindPipeline(c.cb,VK_PIPELINE_BIND_POINT_COMPUTE,p.p);vkCmdBindDescriptorSets(c.cb,VK_PIPELINE_BIND_POINT_COMPUTE,p.l,0,1,&c.ds,0,0);vkCmdPushConstants(c.cb,p.l,VK_SHADER_STAGE_COMPUTE_BIT,0,pc.size()*4,pc.data());vkCmdDispatch(c.cb,(N+BLOCK_SIZE-1)/BLOCK_SIZE,1,1);VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(c.cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0);}
void endC(const VCtx&c){VK_CHECK(vkEndCommandBuffer(c.cb));}
void subOne(const VCtx&c){VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&c.cb;VK_CHECK(vkQueueSubmit(c.q,1,&si,c.fence));}
void waitOne(const VCtx&c){VK_CHECK(vkWaitForFences(c.dev,1,&c.fence,VK_TRUE,UINT64_MAX));VK_CHECK(vkResetFences(c.dev,1,&c.fence));}
void precompute_rope(float*cos,float*sin,int ml,int dk){int h=dk/2;std::vector<float>fr(h);for(int i=0;i<h;i++)fr[i]=1.0f/powf(10000.0f,(float)(2*i)/(float)dk);for(int t=0;t<ml;t++)for(int j=0;j<h;j++){float th=(float)t*fr[j];cos[t*h+j]=cosf(th);sin[t*h+j]=sinf(th);}}

struct Qwen3Fwd{
    int V;size_t np;std::vector<size_t> po;size_t mo,vo;
    size_t o_enc,o_rms1,o_q,o_k,o_v,o_qr,o_kr;
    size_t o_rc,o_rs,o_ay,o_at,o_ap,o_rms2,o_sg,o_r3;
    size_t o_rf,o_lo,o_inpi;
};

std::vector<size_t> cps3(int V){return{(size_t)V*C,(size_t)L*C,(size_t)L*NH*DK*C,(size_t)L*NKV*DK*C,(size_t)L*NKV*DK*C,(size_t)L*DK,(size_t)L*DK,(size_t)L*C*C,(size_t)L*C,(size_t)L*DFF*C,(size_t)L*DFF*C,(size_t)L*C*DFF,(size_t)C,(size_t)V*C};}

size_t ga3f(Qwen3Fwd*m,int V,int Bv){
m->V=V;auto s=cps3(V);m->np=0;for(auto x:s)m->np+=x;m->po.resize(s.size());size_t o=0;for(size_t i=0;i<s.size();i++){m->po[i]=o;o+=s[i];}m->mo=o;o+=m->np;m->vo=o;o+=m->np;
auto a=[&](size_t sz){size_t r=o;o+=sz;return r;};
int BT=Bv*T,BTC=Bv*T*C,BTNHDK=Bv*T*NH*DK,BTNKVDK=Bv*T*NKV*DK,BTDFF=Bv*T*DFF,BNHTT=Bv*NH*T*T;
int TDH=T*(DK/2);
m->o_enc=a(BTC);m->o_rms1=a(L*BTC);
m->o_q=a(L*BTNHDK);m->o_k=a(L*BTNKVDK);m->o_v=a(L*BTNKVDK);
m->o_qr=a(L*BTNHDK);m->o_kr=a(L*BTNKVDK);
m->o_rc=a(TDH);m->o_rs=a(TDH);
m->o_ay=a(L*BTC);m->o_at=a(L*BNHTT);m->o_ap=a(L*BTC);
m->o_rms2=a(L*BTC);
m->o_sg=a(L*BTDFF);m->o_r3=a(L*BTC);
m->o_rf=a(BTC);m->o_lo=a(Bv*T*V);
m->o_inpi=0;return o;}

using Clock=std::chrono::high_resolution_clock;

// Pipe arrays indexed by GPU
struct Pipelines{
    std::array<Pipe,MAX_GPU> enc,rms,mm,qkv,qnr,knr,ga,rr,gus,dr;
};

double runBench(int g, int B, int V, VCtx* cx, const Pipelines& pipes){
    Qwen3Fwd md;size_t tf=ga3f(&md,V,B);size_t tib=2*B*T;
    Buf fb_stage,ib_stage,fb_local,ib_local;
    mkBuf(cx[g],tf*4,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,fb_stage);
    mkBuf(cx[g],tib*4,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,ib_stage);
    mkBuf(cx[g],tf*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,fb_local);
    mkBuf(cx[g],tib*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,ib_local);
    memset(fb_stage.mp,0,tf*4);memset(ib_stage.mp,0,tib*4);
    std::mt19937 gn(1337);std::normal_distribution<float>nd(0,0.02);
    float*fp=(float*)fb_stage.mp;for(size_t i=0;i<md.np;i++)fp[i]=nd(gn);
    for(int l=0;l<L;l++)for(size_t i=0;i<C;i++)fp[md.po[1]+l*C+i]=1.0f;
    for(size_t i=0;i<C;i++)fp[md.po[12]+i]=1.0f;
    std::vector<float>hc(T*(DK/2)),hs(T*(DK/2));precompute_rope(hc.data(),hs.data(),T,DK);
    memcpy(fp+md.o_rc,hc.data(),hc.size()*4);memcpy(fp+md.o_rs,hs.data(),hs.size()*4);
    int*ip=(int*)ib_stage.mp;for(int i=0;i<B*T;i++)ip[md.o_inpi+i]=rand()%V;
    
    beginC(cx[g]);
    VkBufferCopy fcopy{0,0,tf*4};VkBufferCopy icopy{0,0,tib*4};
    vkCmdCopyBuffer(cx[g].cb,fb_stage.b,fb_local.b,1,&fcopy);
    vkCmdCopyBuffer(cx[g].cb,ib_stage.b,ib_local.b,1,&icopy);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cx[g].cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,0,0,0);
    endC(cx[g]);subOne(cx[g]);waitOne(cx[g]);
    
    bindBufs(cx[g],fb_local,ib_local);
    
    int BlT=B*T,BlTC=B*T*C,BlTNHDK=B*T*NH*DK,BlTNKVDK=B*T*NKV*DK,BlTDFF=B*T*DFF,BlNHTT=B*NH*T*T;
    auto pp=md.po;
    beginC(cx[g]);
    recC(cx[g],pipes.enc[g],BlTC,{(int)md.o_enc,(int)md.o_inpi,(int)pp[0],B,T,C,0,0,0,0,0,0,0,0,0,0});
    for(int l=0;l<L;l++){
        size_t r=(l==0)?md.o_enc:md.o_r3+(l-1)*BlTC;
        recC(cx[g],pipes.rms[g],BlTC,{(int)(md.o_rms1+l*BlTC),0,(int)r,(int)(pp[1]+l*C),B,T,C,0,0,0,0,0,0,0,0,0});
        recC(cx[g],pipes.qkv[g],BlT*(NH+2*NKV)*DK,{(int)(md.o_q+l*BlTNHDK),(int)(md.o_k+l*BlTNKVDK),(int)(md.o_v+l*BlTNKVDK),(int)(md.o_rms1+l*BlTC),(int)(pp[2]+l*NH*DK*C),(int)(pp[3]+l*NKV*DK*C),(int)(pp[4]+l*NKV*DK*C),B,T,C,NH,NKV,DK,0,0,0});
        recC(cx[g],pipes.qnr[g],BlT*NH*DK,{(int)(md.o_qr+l*BlTNHDK),(int)(md.o_q+l*BlTNHDK),(int)(pp[5]+l*DK),(int)md.o_rc,(int)md.o_rs,B,T,NH,DK,0,0,0,0,0,0,0});
        recC(cx[g],pipes.knr[g],BlT*NKV*DK,{(int)(md.o_kr+l*BlTNKVDK),(int)(md.o_k+l*BlTNKVDK),(int)(pp[6]+l*DK),(int)md.o_rc,(int)md.o_rs,B,T,NKV,DK,0,0,0,0,0,0,0});
        recC(cx[g],pipes.ga[g],BlT*NH*DK,{(int)(md.o_ay+l*BlTC),(int)(md.o_at+l*BlNHTT),(int)(md.o_qr+l*BlTNHDK),(int)(md.o_kr+l*BlTNKVDK),(int)(md.o_v+l*BlTNKVDK),B,T,NH,NKV,DK,NGRP,0,0,0,0,0});
        recC(cx[g],pipes.mm[g],BlTC,{(int)(md.o_ap+l*BlTC),(int)(md.o_ay+l*BlTC),(int)(pp[7]+l*C*C),0,BlT,C,C,0,0,0,0,0,0,0,0});
        recC(cx[g],pipes.rr[g],BlTC,{(int)(md.o_rms2+l*BlTC),(int)r,(int)(md.o_ap+l*BlTC),(int)(pp[8]+l*C),B,T,C,0,0,0,0,0,0,0,0,0});
        recC(cx[g],pipes.gus[g],BlTDFF,{(int)(md.o_sg+l*BlTDFF),(int)(md.o_rms2+l*BlTC),(int)(pp[9]+l*DFF*C),(int)(pp[10]+l*DFF*C),B,T,C,DFF,0,0,0,0,0,0,0});
        recC(cx[g],pipes.dr[g],BlTC,{(int)(md.o_r3+l*BlTC),(int)(md.o_sg+l*BlTDFF),(int)(pp[11]+l*C*DFF),(int)r,(int)(md.o_ap+l*BlTC),B,T,C,DFF,0,0,0,0,0,0,0});
    }
    size_t r=md.o_r3+(L-1)*BlTC;
    recC(cx[g],pipes.rms[g],BlTC,{(int)md.o_rf,0,(int)r,(int)pp[12],B,T,C,0,0,0,0,0,0,0,0,0});
    recC(cx[g],pipes.mm[g],BlT*V,{(int)md.o_lo,(int)md.o_rf,(int)pp[13],0,BlT,C,V,0,0,0,0,0,0,0,0});
    endC(cx[g]);
    
    subOne(cx[g]);waitOne(cx[g]);
    for(int w=0;w<3;w++){subOne(cx[g]);waitOne(cx[g]);}
    auto t0=Clock::now();for(int r2=0;r2<10;r2++){subOne(cx[g]);waitOne(cx[g]);}
    auto t1=Clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/10;
    
    vkDeviceWaitIdle(cx[g].dev);
    destroyBuf(cx[g],fb_stage);destroyBuf(cx[g],ib_stage);
    destroyBuf(cx[g],fb_local);destroyBuf(cx[g],ib_local);
    return ms;
}

int main(){
srand(1337);
std::ifstream file("input.txt");
int V=32; 
if(file.is_open()){std::string text((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());file.close();
struct CharTokenizer{std::map<char,int>stoi;std::map<int,char>itos;int vocab_size;
void init(const std::string&t){std::vector<char>u;for(char c:t)if(stoi.find(c)==stoi.end()){stoi[c]=(int)u.size();itos[(int)u.size()]=c;u.push_back(c);}std::sort(u.begin(),u.end());stoi.clear();itos.clear();for(int i=0;i<(int)u.size();i++){stoi[u[i]]=i;itos[i]=u[i];}vocab_size=(int)u.size();}};CharTokenizer tok;tok.init(text);V=tok.vocab_size;}
printf("V: %d\n",V);

uint32_t dc=0;VkInstance inst;
{VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};ai.pApplicationName="Bench";ai.apiVersion=VK_API_VERSION_1_2;VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ci.pApplicationInfo=&ai;VK_CHECK(vkCreateInstance(&ci,0,&inst));}
VK_CHECK(vkEnumeratePhysicalDevices(inst,&dc,0));std::vector<VkPhysicalDevice>pds(dc);VK_CHECK(vkEnumeratePhysicalDevices(inst,&dc,pds.data()));
int ng=std::min((int)dc,MAX_GPU);VCtx cx[MAX_GPU];
for(int g=0;g<ng;g++){initCtx(cx[g],inst,pds[g]);VkPhysicalDeviceProperties pr;vkGetPhysicalDeviceProperties(pds[g],&pr);printf("  GPU %d: %s\n",g,pr.deviceName);}

// Create pipelines for all GPUs
Pipelines pipes;
auto mkP=[&](const char*src,const char*nm)->std::array<Pipe,MAX_GPU>{std::array<Pipe,MAX_GPU>a;auto spv=comp(src,nm);for(int g=0;g<ng;g++)a[g]=mkPipe(cx[g],mkSM(cx[g],spv));return a;};
pipes.enc=mkP(sh_enc,"enc");pipes.rms=mkP(sh_rms,"rms");pipes.mm=mkP(sh_mm,"mm");
pipes.qkv=mkP(sh_qkv,"qkv");pipes.qnr=mkP(sh_q_norm_rope,"qnr");pipes.knr=mkP(sh_q_norm_rope,"knr");
pipes.ga=mkP(sh_gqa_v2,"gqa2");pipes.rr=mkP(sh_res_rms,"rr");
pipes.gus=mkP(sh_gate_up_swiglu,"gus");pipes.dr=mkP(sh_down_res,"dr");

int test_B[]={1,2,4,8,12,16,20,24,32,48,64,96,128,192,256};
int num_tests=sizeof(test_B)/sizeof(test_B[0]);

printf("\n=== QWEN3 FORWARD PASS BENCHMARK V4 (3-GPU, DEVICE-LOCAL) ===\n");
printf("Model: C=%d NH=%d NKV=%d L=%d T=%d DFF=%d V=%d\n",C,NH,NKV,L,T,DFF,V);

printf("%-6s %-12s %-12s %-12s %-12s\n","GPU","B","ms/fwd","samples/s","ms/sample");
printf("------ ------------ ------------ ------------ ------------\n");

// Collect data per GPU
std::map<int,double> ms_per_sample[MAX_GPU];

for(int g=0;g<ng;g++){
    for(int ti=0;ti<num_tests;ti++){
        int B=test_B[ti];
        double ms;
        printf("%-6d %-12d ",g,B);fflush(stdout);
        try {
            ms=runBench(g,B,V,cx,pipes);
        } catch(...){
            printf("OOM/CRASH\n");
            ms_per_sample[g][B]=9999.0;
            continue;
        }
        double mps=ms/B;
        ms_per_sample[g][B]=mps;
        printf("%-12.1f %-12.0f %-12.2f\n",ms,B/(ms/1000.0),mps);
    }
    printf("\n");
}

// Interpolation helper
auto interp = [&](int g, int B) -> double {
    auto& m = ms_per_sample[g];
    if(B <= m.begin()->first) return m.begin()->second;
    if(B >= m.rbegin()->first) return m.rbegin()->second;
    auto hi = m.lower_bound(B);
    if(hi->first == B) return hi->second;
    auto lo = std::prev(hi);
    double t = (double)(B - lo->first) / (hi->first - lo->first);
    return lo->second + t * (hi->second - lo->second);
};

printf("\n=== MULTI-GPU OPTIMAL SPLIT ANALYSIS ===\n\n");

// Test various total batch sizes
for(int total_B : {16, 32, 64, 96, 128, 192, 256}){
    printf("━━━ B_TOTAL = %d ━━━\n",total_B);
    
    // Single GPU baseline (fastest GPU)
    int best_single = 0;
    double best_single_time = 1e18;
    for(int g=0;g<ng;g++){
        double t = total_B * interp(g, total_B);
        if(t < best_single_time){ best_single_time = t; best_single = g; }
    }
    printf("  Best single GPU: GPU%d @ %.1f ms\n",best_single,best_single_time);
    
    // 2-GPU combos
    for(int g0=0;g0<ng;g0++){
        for(int g1=g0+1;g1<ng;g1++){
            double best2=1e18;int best_B0=1;
            for(int B0=1;B0<total_B;B0++){
                int B1=total_B-B0;
                double t0=B0*interp(g0,B0),t1=B1*interp(g1,B1);
                double t=std::max(t0,t1);
                if(t<best2){best2=t;best_B0=B0;}
            }
            int B1=total_B-best_B0;
            double t0=best_B0*interp(g0,best_B0),t1=B1*interp(g1,B1);
            printf("  GPU%d=%d GPU%d=%d: %.1f ms (%.2fx, bal: %.1f/%.1f)\n",
                   g0,best_B0,g1,B1,best2,best_single_time/best2,t0,t1);
        }
    }
    
    // 3-GPU combo
    if(ng>=3){
        double best3=1e18;int best_B0=1,best_B1=1;
        for(int B0=1;B0<total_B-1;B0++){
            for(int B1=1;B1<total_B-B0;B1++){
                int B2=total_B-B0-B1;
                double t0=B0*interp(0,B0),t1=B1*interp(1,B1),t2=B2*interp(2,B2);
                double t=std::max({t0,t1,t2});
                if(t<best3){best3=t;best_B0=B0;best_B1=B1;}
            }
        }
        int B2=total_B-best_B0-best_B1;
        double t0=best_B0*interp(0,best_B0),t1=best_B1*interp(1,best_B1),t2=B2*interp(2,B2);
        printf("  GPU0=%d GPU1=%d GPU2=%d: %.1f ms (%.2fx, bal: %.1f/%.1f/%.1f)\n",
               best_B0,best_B1,B2,best3,best_single_time/best3,t0,t1,t2);
    }
    printf("\n");
}

return 0;}