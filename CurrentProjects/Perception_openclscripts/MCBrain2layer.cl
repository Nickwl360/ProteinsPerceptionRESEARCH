 #define MAXTOP 5
#define MAXBOT 12
#define M1 MAXTOP*MAXBOT*MAXBOT*MAXTOP*MAXTOP*MAXBOT*MAXBOT
#define M2 M1/MAXTOP
#define M3 M2/MAXBOT
#define M4 M3/MAXBOT
#define M5 M4/MAXTOP
#define M6 M5/MAXTOP
#define PI 3.1415
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


__kernel void compute_Pmnop(__global double* Pmnop, const double halpha, const double hbeta, const double hgamma, const double hdelta,const double ha,const double hb, const double hc, const double hd, const double kcoop,const double kcomp,const double kdu, const double kup,const double kx)
{
     int i = get_global_id(0);   //ARRAYindexes:[nai,nbj,nck,ndl, nam, nbn, nco, ndp] [i,j,k,l ,m,n,o,p]

     int nai = (i/(M1));
     int nbj = (i/(M2))%MAXTOP;
     int nck =(i/(M3))%MAXBOT;
     int ndl = (i/(M4))%MAXBOT;
     int nam = (i/(M5))%MAXTOP;
     int nbn = (i/(M6))%MAXTOP;
     int nco = (i/(MAXBOT))%MAXBOT;
     int ndp = (i)%MAXBOT;


     double loop4=0;   //a=lA, b=lB, c=lC,d=lD, e=la,f=lb,g=lc,h=ld
     for(int a=0; a<=nai;a++)
     {
        for(int b=0; b<=nbj;b++)
        {
            for (int c=0; c<=M;c++)
            {
                for(int d=0; d<=M;d++)
                {
                    int e = nam - nai + a;  //  lalpha
                    int f = nbn - nbj + b;  //lbeta
                    int g = nco - nck + c;  //lgamma
                    int h = ndp - ndl + d; // ldelta
                    //if(0 <= e && e <= (MAXTOP-1-nai) && 0 <= f && f <= (MAXTOP-1-nbj) && 0 <= g && g <= (MAXBOT-1-nck) && 0 <= h && h <= (MAXBOT-1-ndl) )
                    if(0 <= e && e <= (M) && 0 <= f && f <= (M) && 0 <= g && g <= (M) && 0 <= h && h <= (M) )

                    {
                        loop4 +=  exp(calcP(nai,nbj,nck,ndl, e,a,f,b,g,c,h,d, halpha,ha,hbeta,hb,hgamma,hdelta,hc,hd,kcoop,kcomp,kdu,kup,kx));
                    }
                }

            }


        }

    }
    Pmnop[i] = loop4;
}


