#define MAXTOP 5
#define MAXBOT 12
#define BI 2
#define PI 3.14159265358979323
#define M 1

double stirling(double x){
    if(x ==0){
        return 0.0;
    }
    else{
        return x * log(x) - x + .5*log(2*PI*x);
    }
}

double newcomb(double N, double L) {
    int a = N;
    int b = L;
    double n = N;
    double l = L;

    if (a == 0) {
        return 0.0;
        }
    if (b == 0 || a == b){
        return (stirling(ceil(n/2)) + stirling(floor(n/2)) - stirling(n));
        }
    else{
        return (stirling(ceil(n/2)) + stirling(floor(n/2)) - stirling(l)- stirling(n-l));
        }
    }

double calcP(double NA, double NB, double NC, double ND, double la,double lA,double lb, double lB, double lgamma, double lc, double ldelta, double ld,double halpha, double hA, double hbeta, double hB, double hgamma,double hdelta, double hc, double hd, double kcoop,double kcomp, double kdu, double kud, double kx){
    double e = (halpha * (la-((MAXTOP-NA)/2)) + hbeta * (lb - ((MAXTOP-NB)/2)) + hgamma*(lgamma-((MAXBOT-NC))/2) + hdelta*(ldelta-(MAXBOT-ND)/2) + hA * (lA-NA/2) + hB*(lB-NB/2) + hc*(lc-NC/2)+hd*(ld-ND/2) + kcoop*((la-lA)*NA - (MAXTOP*NA/2) + (lb-lB)*NB- (MAXTOP*NB/2)) + kcomp*((lA-la)*NB -(MAXTOP*NB/2) + (lB-lb)*NA - (MAXTOP*NA/2)) + kdu*((la-lA)*NC - (NC*MAXTOP/2) +(lb-lB)*ND -(ND*MAXTOP/2)) + kud*(NA*(lc-lgamma)-(NA*MAXBOT/2)+NB*(ld-ldelta)-(NB*MAXBOT/2))+kx*((lB-lb)*NC-(NC*MAXTOP/2)+(lA-la)*ND-(ND*MAXTOP/2)));
    double sum = (newcomb(MAXTOP-1-NA,la) + newcomb(MAXTOP-1-NB,lb)+ newcomb(NA, lA) + newcomb(NB, lB) + newcomb(MAXBOT-1-NC,lgamma) + newcomb(NC,lc) + newcomb(MAXBOT-1-ND,ldelta) + newcomb(ND,ld) + e);
    return sum;
}


__kernel void compute_Pls(__global double* Pls, const double halpha, const double hbeta, const double hgamma, const double hdelta,const double ha,const double hb, const double hc, const double hd, const double kcoop,const double kcomp,const double kdu, const double kud,const double kx)
{
     int i = get_global_id(0);   //ARRAYindexes:[nai,nbj,nck,ndl, ll+, ll-, lu+, ll+',ll-',lu+', lu-,lu-'] [i,j,k,l ,m,n,o,p]

     int NA = (i/(MAXTOP*MAXBOT*MAXBOT*BI*BI*BI*BI*BI*BI*MAXTOP*MAXTOP));
     int NB = (i/(MAXBOT*MAXBOT*BI*BI*BI*BI*BI*BI*MAXTOP*MAXTOP))%MAXTOP;
     int NC = (i/(MAXBOT*BI*BI*BI*BI*BI*BI*MAXTOP*MAXTOP))%MAXBOT;
     int ND = (i/(BI*BI*BI*BI*BI*BI*MAXTOP*MAXTOP))%MAXBOT;
     int llp = (i/(BI*BI*BI*BI*BI*MAXTOP*MAXTOP))%BI;
     int llm = (i/(BI*BI*BI*BI*MAXTOP*MAXTOP))%BI;
     int lup = (i/(BI*BI*BI*MAXTOP*MAXTOP))%BI;
     int llp_ = (i/(BI*BI*MAXTOP*MAXTOP))%BI;
     int llm_ = (i/(BI*MAXTOP*MAXTOP))%BI;
     int lup_ = (i/(MAXTOP*MAXTOP))%BI;
     int lum = (i/(MAXTOP))%MAXTOP;
     int lum_ = (i)%MAXTOP;


     if((lum<=NA) && (lum_ <= NB)&& (llm<=NC)&&(llm_<=ND) && (llp <= MAXBOT-1-NC)&& (llp_ <= MAXBOT-1-ND) && (lup<=MAXTOP-1-NA) && (lup_<=MAXTOP-1-NB))
     {
        Pls[i]= exp(calcP(NA,NB,NC,ND,lup,lum,lup_,lum_,llp,llm,llp_,llm_,halpha,ha,hbeta,hb,hgamma,hdelta,hc,hd,kcoop,kcomp,kdu,kud,kx));
     }
     //&& (llm<=NC) && (llm_<=ND) && (lup <= (NA-MAXTOP-1)) && (lup_<=(NB-MAXTOP-1)) && (llp<= (NC-MAXBOT-1)) && (llp_ <=(ND-MAXBOT-1)))
}


